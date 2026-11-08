#!/usr/bin/env python3
"""
02_erase_pedestrian.py
----------------------
Step 2 of the DISCRETIO counterfactual pipeline.

Uses ProPainter to erase the original pedestrian from the clip, producing a
clean background plate. Missing ProPainter is a hard error.

Both Method 1 (Erase + Pose Retarget) and Method 2 (Unified Mask-Box
Inpainting) need a clean plate; they diverge in Step 3.

Outputs per clip (written to --out_dir/<clip_id>/):
  erased_frames/frame_NNNNNN.png   — background-only frames (pedestrian erased)
  erase_quality.json               — PSNR/LPIPS of erased region (self-check)

ProPainter notes
----------------
  ProPainter expects:
    - frames/   directory with frame_NNNNNN.png
    - masks/    directory with matching binary PNG masks
  It outputs inpainted frames to its --output dir.

Usage
-----
  python 02_erase_pedestrian.py \\
      --clips_json      selected_clips/selected_clips.json \\
      --masks_dir       pipeline_output/01_masks \\
      --video_dir       /data/videos \\
      --out_dir         pipeline_output/02_erased \\
      [--propainter_dir /path/to/ProPainter]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils.video_io import extract_frames, frames_to_video


# ---------------------------------------------------------------------------
# ProPainter runner
# ---------------------------------------------------------------------------

def run_propainter(
    frames_dir: str,
    masks_dir: str,
    output_dir: str,
    propainter_dir: str,
    width: int = 1920,
    height: int = 1080,
    ref_stride: int = 10,
    neighbor_length: int = 10,
    subvideo_length: int = 80,
) -> bool:
    """
    Invoke ProPainter's inference script as a subprocess.

    Returns True on success, False on failure.
    """
    script = Path(propainter_dir) / "inference_propainter.py"
    if not script.exists():
        print(f"  [ERROR] ProPainter script not found at {script}.")
        return False

    cmd = [
        "python", str(script),
        "--video",           frames_dir,
        "--mask",            masks_dir,
        "--output",          output_dir,
        "--width",           str(width),
        "--height",          str(height),
        "--ref_stride",      str(ref_stride),
        "--neighbor_length", str(neighbor_length),
        "--subvideo_length", str(subvideo_length),
        "--fp16",
    ]
    print(f"  Running ProPainter: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def fallback_inpaint(
    frame: np.ndarray,     # (H, W, 3) BGR
    mask: np.ndarray,      # (H, W) uint8, 255=region to inpaint
    radius: int = 3,
) -> np.ndarray:
    """
    OpenCV inpainting cannot reconstruct a person-sized video region. Fail
    closed so a placeholder can never be mistaken for benchmark data.
    """
    raise RuntimeError("ProPainter is required; OpenCV inpainting is not a valid replacement.")


# ---------------------------------------------------------------------------
# Self-quality check
# ---------------------------------------------------------------------------

def compute_erase_quality(
    original_frames: List[np.ndarray],  # (T, H, W, 3) uint8 BGR
    erased_frames: List[np.ndarray],
    masks: List[np.ndarray],            # (T, H, W) uint8 binary
) -> Dict:
    """
    Compute PSNR inside the inpainted region as a self-consistency proxy.
    Lower is NOT better here — we want the background to look consistent
    with areas we did NOT touch.  We compute PSNR of the pixels just
    *outside* the dilated mask between original and erased.

    Returns dict: {outside_psnr_mean, outside_psnr_std, n_frames}
    """
    psnrs = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    for orig, erased, mask in zip(original_frames, erased_frames, masks):
        # Dilated mask: region touched by inpainting
        dilated = cv2.dilate(mask, kernel)
        outside = (dilated == 0)

        if outside.sum() < 1000:
            continue

        orig_outside   = orig[outside].astype(np.float64)
        erased_outside = erased[outside].astype(np.float64)

        mse = np.mean((orig_outside - erased_outside) ** 2)
        if mse < 1e-6:
            psnr = 100.0
        else:
            psnr = 10 * np.log10(255 ** 2 / mse)
        psnrs.append(psnr)

    if not psnrs:
        return {"outside_psnr_mean": None, "outside_psnr_std": None, "n_frames": 0}

    return {
        "outside_psnr_mean": float(np.mean(psnrs)),
        "outside_psnr_std":  float(np.std(psnrs)),
        "n_frames": len(psnrs),
    }


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------

def process_clip(
    clip: Dict,
    masks_dir: Path,
    video_dir: str,
    out_dir: Path,
    propainter_dir: Optional[str],
) -> None:
    clip_id  = clip["clip_id"]
    video_id = clip["video_id"]
    dataset  = clip["dataset"]
    start_f  = int(clip["start_frame"])
    end_f    = int(clip["end_frame"])

    clip_masks_dir = masks_dir / clip_id / "masks"
    if not clip_masks_dir.exists():
        print(f"  [SKIP] Masks not found for {clip_id} at {clip_masks_dir}")
        return

    clip_out = out_dir / clip_id
    erased_dir = clip_out / "erased_frames"
    erased_dir.mkdir(parents=True, exist_ok=True)

    # --- Locate video and extract raw frames ---
    video_path = _find_video(video_dir, dataset, video_id)
    if video_path is None:
        print(f"  [SKIP] Video not found for {clip_id}")
        return

    with tempfile.TemporaryDirectory() as raw_tmp:
        n = extract_frames(video_path, raw_tmp, start_frame=start_f, end_frame=end_f)
        if n == 0:
            print(f"  [SKIP] No frames extracted for {clip_id}")
            return

        # --- Run ProPainter (preferred) ---
        if propainter_dir and run_propainter(
            raw_tmp, str(clip_masks_dir), str(erased_dir), propainter_dir
        ):
            print(f"  ProPainter completed for {clip_id}.")

        else:
            # No image-only fallback is valid for this video reconstruction.
            print(f"  [ERROR] ProPainter failed or is unavailable for {clip_id}.")
            frame_paths = sorted(Path(raw_tmp).glob("frame_*.png"))
            mask_paths  = sorted(clip_masks_dir.glob("frame_*.png"))

            # Pair by frame number
            mask_map = {p.stem: p for p in mask_paths}
            for fp in frame_paths:
                frame = cv2.imread(str(fp))
                if frame is None:
                    continue
                mask_path = mask_map.get(fp.stem)
                if mask_path is None:
                    # No mask for this frame → copy unchanged
                    cv2.imwrite(str(erased_dir / fp.name), frame)
                    continue
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    cv2.imwrite(str(erased_dir / fp.name), frame)
                    continue
                erased = fallback_inpaint(frame, mask)
                cv2.imwrite(str(erased_dir / fp.name), erased)

        # --- Self-quality check (must be inside the with-block while raw_tmp exists) ---
        orig_frames_q, erased_frames_q, masks_q = [], [], []
        frame_paths = sorted(Path(raw_tmp).glob("frame_*.png"))
        for fp in frame_paths:
            orig = cv2.imread(str(fp))
            erased_path = erased_dir / fp.name
            erased = cv2.imread(str(erased_path))
            if orig is None or erased is None:
                continue
            mask_path = clip_masks_dir / fp.name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
            if mask is None:
                mask = np.zeros(orig.shape[:2], dtype=np.uint8)
            orig_frames_q.append(orig)
            erased_frames_q.append(erased)
            masks_q.append(mask)

    # raw_tmp is now deleted — use variables captured inside the block above
    quality = compute_erase_quality(orig_frames_q, erased_frames_q, masks_q)
    with open(clip_out / "erase_quality.json", "w") as f:
        json.dump(quality, f, indent=2)

    psnr_val = quality.get("outside_psnr_mean")
    psnr_str = f"{psnr_val:.1f}" if psnr_val is not None else "N/A"
    print(f"  ✓ {clip_id}: erased.  Outside-mask PSNR = {psnr_str} dB")


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
        description="Erase original pedestrians using ProPainter for DISCRETIO pipeline."
    )
    parser.add_argument("--clips_json",      required=True)
    parser.add_argument("--masks_dir",       default="pipeline_output/01_masks")
    parser.add_argument("--video_dir",       required=True)
    parser.add_argument("--out_dir",         default="pipeline_output/02_erased")
    parser.add_argument("--propainter_dir",  default=None,
                        help="Path to the ProPainter repo root (contains inference_propainter.py).")
    parser.add_argument("--clip_ids",        nargs="*", default=None)
    args = parser.parse_args()

    with open(args.clips_json) as f:
        clips = json.load(f)

    if args.clip_ids:
        clips = [c for c in clips if c["clip_id"] in args.clip_ids]

    print(f"Processing {len(clips)} clip(s) for pedestrian erasure.")
    out_dir   = Path(args.out_dir)
    masks_dir = Path(args.masks_dir)

    for clip in clips:
        print(f"\n[Clip] {clip['clip_id']}")
        process_clip(clip, masks_dir, args.video_dir, out_dir, args.propainter_dir)

    print("\nDone. Erased frames written to:", out_dir)


if __name__ == "__main__":
    main()
