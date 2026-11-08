#!/usr/bin/env python3
"""
03b_unified_inpaint.py
-----------------------
Step 3b — Method 2: Unified Mask-Box Inpainting.

Instead of separately erasing then re-inserting a posed pedestrian, this
method expands the SAM2 bounding box and feeds the original frame + expanded
mask directly to Wan2.1-VACE 14B for unified inpainting.

Wan2.1-VACE sees the original context (clothing, lighting, motion blur)
and replaces only the masked region with a pedestrian of the target age
bracket — producing higher visual realism at the cost of slightly weaker
factor isolation (the model must also infer posture from context).

Key design choices
------------------
  - Expanded BBox scale: 1.4× (default) to give the model room to generate
    a differently-proportioned actor.
  - Actor-specific aspect ratio priors (from utils/bbox_ops.ACTOR_ASPECT_RATIOS)
    shape the expansion so the generated person fits naturally.
  - PSNR/LPIPS is computed on the region OUTSIDE the expanded mask (§Metrics)
    to verify background preservation.

Output per clip (--out_dir/<clip_id>/method2/):
  frames/frame_NNNNNN.png   — final inpainted frames
  metadata.json             — target_age, bbox_scale, background PSNR/LPIPS

Usage
-----
  python 03b_unified_inpaint.py \\
      --clips_json      selected_clips/selected_clips.json \\
      --masks_dir       pipeline_output/01_masks \\
      --video_dir       /data/videos \\
      --out_dir         pipeline_output/03b_method2 \\
      --target_age      child \\
      [--wan_checkpoint /path/to/Wan2.1-VACE-14B] \\
      [--bbox_scale     1.4]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils.bbox_ops import (
    expand_bbox,
    bbox_to_mask,
    ACTOR_ASPECT_RATIOS,
)
from utils.video_io import extract_frames

# ---------------------------------------------------------------------------
# Text prompts for unified inpainting
# ---------------------------------------------------------------------------
AGE_PROMPTS = {
    "child":  ("a young child pedestrian, about 8 years old, same pose, "
               "same lighting, seamlessly blended into the scene, photorealistic"),
    "adult":  ("an adult pedestrian, same pose and clothing style, "
               "same lighting, seamlessly blended, photorealistic"),
    "senior": ("an elderly pedestrian, about 70 years old, same pose, "
               "same lighting, seamlessly blended into the scene, photorealistic"),
}

NEGATIVE_PROMPT = (
    "blurry, artifacts, deformed, unrealistic background changes, "
    "extra limbs, duplicate person, watermark"
)


# ---------------------------------------------------------------------------
# Wan2.1-VACE unified inpaint runner
# ---------------------------------------------------------------------------

def run_wan_unified_inpaint(
    video_frames_dir: str,
    mask_frames_dir: str,
    output_dir: str,
    prompt: str,
    wan_checkpoint: str,
    sample_steps: int = 50,
) -> bool:
    """
    Invoke Wan2.1-VACE for mask-conditioned video inpainting.

    Wan2.1-VACE in inpainting mode takes:
      --video       : directory of original frames
      --mask_video  : directory of binary mask frames (white = inpaint region)
      --prompt      : text conditioning
      --output_dir  : where to write generated frames

    Returns True on success, False if checkpoint not found.
    """
    wan_script = Path(wan_checkpoint) / "generate.py"
    if not wan_script.exists():
        raise FileNotFoundError(f"Wan checkpoint/script not found: {wan_script}")

    cmd = [
        "python", str(wan_script),
        "--task",               "i2v-14B",
        "--mode",               "inpaint",
        "--prompt",             prompt,
        "--negative_prompt",    NEGATIVE_PROMPT,
        "--video",              video_frames_dir,
        "--mask_video",         mask_frames_dir,
        "--output_dir",         output_dir,
        "--ckpt_dir",           wan_checkpoint,
        "--sample_steps",       str(sample_steps),
        "--sample_guide_scale", "5.5",
        "--fp16",
    ]
    print(f"  Wan2.1-VACE unified inpaint: {' '.join(cmd[:6])} ...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Wan generation failed (exit {result.returncode}); see Wan output above.")
    return True


# ---------------------------------------------------------------------------
# Background quality metrics (outside the inpainted region)
# ---------------------------------------------------------------------------

def compute_background_metrics(
    orig_frames_dir: str,
    edited_frames_dir: str,
    expanded_mask_dir: str,
) -> Dict:
    """
    Compute PSNR and LPIPS on pixels OUTSIDE the expanded inpaint mask.
    Background should be near-identical between original and edited clip.

    Returns: {psnr_mean, psnr_std, lpips_mean, n_frames}
    """
    psnrs: List[float] = []
    lpips_scores: List[float] = []
    lpips_fn = None

    try:
        import lpips  # type: ignore
        import torch
        lpips_fn = lpips.LPIPS(net="alex").eval()
    except ImportError:
        pass

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    for p in sorted(Path(orig_frames_dir).glob("frame_*.png")):
        orig   = cv2.imread(str(p))
        edited = cv2.imread(str(Path(edited_frames_dir) / p.name))
        mask_p = Path(expanded_mask_dir) / p.name
        mask   = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) if mask_p.exists() else None

        if orig is None or edited is None:
            continue

        if mask is None:
            mask = np.zeros(orig.shape[:2], dtype=np.uint8)

        # Dilate mask to exclude border pixels
        dilated = cv2.dilate(mask, kernel)
        outside = dilated == 0

        if outside.sum() < 1000:
            continue

        # PSNR
        orig_outside   = orig[outside].astype(np.float64)
        edited_outside = edited[outside].astype(np.float64)
        mse = np.mean((orig_outside - edited_outside) ** 2)
        psnr = 100.0 if mse < 1e-6 else 10 * np.log10(255 ** 2 / mse)
        psnrs.append(psnr)

        # LPIPS (patch-level on background region only)
        if lpips_fn is not None:
            try:
                import torch
                h, w = orig.shape[:2]
                # Use random 256×256 background patch for speed
                ys, xs = np.where(outside)
                if len(ys) > 0:
                    idx = np.random.randint(len(ys))
                    y0 = max(0, ys[idx] - 128); y1 = min(h, y0 + 256)
                    x0 = max(0, xs[idx] - 128); x1 = min(w, x0 + 256)
                    if (y1 - y0) >= 64 and (x1 - x0) >= 64:
                        p_orig = _to_lpips_tensor(orig[y0:y1, x0:x1])
                        p_edit = _to_lpips_tensor(edited[y0:y1, x0:x1])
                        with torch.no_grad():
                            score = float(lpips_fn(p_orig, p_edit).item())
                        lpips_scores.append(score)
            except Exception:
                pass

    result: Dict = {
        "psnr_mean":  float(np.mean(psnrs))  if psnrs  else None,
        "psnr_std":   float(np.std(psnrs))   if psnrs  else None,
        "lpips_mean": float(np.mean(lpips_scores)) if lpips_scores else None,
        "n_frames":   len(psnrs),
    }
    return result


def _to_lpips_tensor(bgr_crop: np.ndarray):
    import torch
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------

def process_clip(
    clip: Dict,
    target_age: str,
    masks_dir: Path,
    video_dir: str,
    out_dir: Path,
    wan_checkpoint: str,
    bbox_scale: float,
    image_width: int,
    image_height: int,
) -> Dict:
    clip_id    = clip["clip_id"]
    dataset    = clip["dataset"]
    video_id   = clip["video_id"]
    start_f    = int(clip["start_frame"])
    end_f      = int(clip["end_frame"])
    source_age = clip["age_bracket"]

    clip_out = out_dir / clip_id / "method2"
    clip_out.mkdir(parents=True, exist_ok=True)

    # Load per-frame bboxes from Step 1
    bb_path = masks_dir / clip_id / "bboxes.json"
    if not bb_path.exists():
        print(f"  [SKIP] bboxes.json not found for {clip_id}")
        return {}
    with open(bb_path) as f:
        bboxes = json.load(f)

    # Locate raw video
    video_path = _find_video(video_dir, dataset, video_id)
    if video_path is None:
        print(f"  [SKIP] Video not found for {clip_id}")
        return {}

    # Extract raw frames
    raw_frames_dir = clip_out / "raw_frames"
    raw_frames_dir.mkdir(exist_ok=True)
    n = extract_frames(video_path, str(raw_frames_dir), start_frame=start_f, end_frame=end_f)
    if n == 0:
        print(f"  [SKIP] No frames extracted for {clip_id}")
        return {}
    first_frame = cv2.imread(str(raw_frames_dir / f"frame_{start_f:06d}.png"))
    if first_frame is None:
        raise RuntimeError(f"Could not read extracted source frame for {clip_id}")
    actual_height, actual_width = first_frame.shape[:2]

    # Build expanded mask frames
    aspect_ratio = ACTOR_ASPECT_RATIOS.get(target_age)
    exp_mask_dir = clip_out / "expanded_masks"
    exp_mask_dir.mkdir(exist_ok=True)

    for fn_str, bbox in bboxes.items():
        fn = int(fn_str)
        expanded = expand_bbox(
            tuple(bbox), scale=bbox_scale,
            image_width=actual_width, image_height=actual_height,
            aspect_ratio=aspect_ratio,
        )
        mask = bbox_to_mask(expanded, actual_width, actual_height)
        cv2.imwrite(str(exp_mask_dir / f"frame_{fn:06d}.png"), mask)

    # Run Wan2.1-VACE unified inpainting
    prompt = AGE_PROMPTS[target_age]
    wan_out_dir = clip_out / "wan_output"
    wan_out_dir.mkdir(exist_ok=True)
    wan_ok = run_wan_unified_inpaint(
        str(raw_frames_dir), str(exp_mask_dir),
        str(wan_out_dir), prompt, wan_checkpoint,
    )
    if not wan_ok:
        raise RuntimeError("Unified inpainting did not complete; refusing placeholder evaluation.")

    # Copy generated frame files to the final evaluation directory.
    final_dir = clip_out / "frames"
    final_dir.mkdir(exist_ok=True)
    for p in sorted(wan_out_dir.glob("frame_*.png")):
        cv2.imwrite(str(final_dir / p.name), cv2.imread(str(p)))

    # Compute background preservation metrics
    metrics = compute_background_metrics(
        str(raw_frames_dir), str(final_dir), str(exp_mask_dir)
    )

    metadata = {
        "clip_id":       clip_id,
        "source_age":    source_age,
        "target_age":    target_age,
        "bbox_scale":    bbox_scale,
        "aspect_ratio":  aspect_ratio,
        "wan_success":   wan_ok,
        **{f"bg_{k}": v for k, v in metrics.items()},
    }
    with open(clip_out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    psnr_m = metrics.get("psnr_mean")
    psnr_str = f"{psnr_m:.1f}" if psnr_m is not None else "N/A"
    lpips_m = metrics.get("lpips_mean")
    lpips_str = f"{lpips_m:.3f}" if lpips_m is not None else "N/A"
    print(f"  ✓ {clip_id} → Method 2 complete.  BG PSNR={psnr_str} dB  LPIPS={lpips_str}")
    return metadata


def _find_video(video_dir: str, dataset: str, video_id: str):
    video_dir = Path(video_dir)
    for ext in (".mp4", ".avi", ".mov", ".MP4"):
        for root in [video_dir, video_dir / dataset]:
            p = root / f"{video_id}{ext}"
            if p.exists():
                return str(p)
            for found in root.glob(f"**/{video_id}{ext}"):
                return str(found)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Method 2: Unified Expanded-BBox Inpainting (Wan2.1-VACE 14B)."
    )
    parser.add_argument("--clips_json",     required=True)
    parser.add_argument("--masks_dir",      default="pipeline_output/01_masks")
    parser.add_argument("--video_dir",      required=True)
    parser.add_argument("--out_dir",        default="pipeline_output/03b_method2")
    parser.add_argument("--target_age",     choices=["child", "adult", "senior"],
                        required=True)
    parser.add_argument("--wan_checkpoint", default="Wan2.1-VACE-14B")
    parser.add_argument("--bbox_scale",     type=float, default=1.4,
                        help="Expansion factor for the inpaint bounding box.")
    parser.add_argument("--image_width",    type=int, default=1920)
    parser.add_argument("--image_height",   type=int, default=1080)
    parser.add_argument("--clip_ids",       nargs="*", default=None)
    args = parser.parse_args()

    with open(args.clips_json) as f:
        clips = json.load(f)
    if args.clip_ids:
        clips = [c for c in clips if c["clip_id"] in args.clip_ids]

    out_dir   = Path(args.out_dir)
    masks_dir = Path(args.masks_dir)

    all_meta = []
    for clip in clips:
        print(f"\n[Clip] {clip['clip_id']}  source_age={clip['age_bracket']}  "
              f"→ target_age={args.target_age}")
        meta = process_clip(
            clip, args.target_age, masks_dir, args.video_dir,
            out_dir, args.wan_checkpoint, args.bbox_scale,
            args.image_width, args.image_height,
        )
        if meta:
            all_meta.append(meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"method2_{args.target_age}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\nMethod 2 summary written to: {summary_path}")


if __name__ == "__main__":
    main()
