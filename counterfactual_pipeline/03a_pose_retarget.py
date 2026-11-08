#!/usr/bin/env python3
"""
03a_pose_retarget.py
--------------------
Step 3a — Method 1: Erase + Retargeted Pose Insertion.

Takes the clean background plate (from Step 2) and re-inserts a pedestrian
of the TARGET age bracket using pose-driven video diffusion (Wan2.1-VACE).

Key principle: the foot-anchor of the new pedestrian is pinned to the
foot-anchor of the ORIGINAL pedestrian at every frame, ensuring the
factor-isolation requirement (position/heading held constant).

Pipeline
--------
  1. Load original bounding boxes + foot anchors (from Step 1).
  2. Build a pose-conditioning sequence:
       a. Extract OpenPose / DWPose skeleton from the ORIGINAL pedestrian crops.
       b. Scale the skeleton to match the target age's height ratio.
          (child ≈ 0.60 × adult height; senior ≈ 0.95 × adult height)
       c. Translate so the foot anchor stays fixed.
  3. Run Wan2.1-VACE with the scaled pose sequence + text prompt describing
     the target age bracket.
  4. Composite the generated pedestrian back onto the erased plate using
     the SAM2 mask (dilated slightly for clean edges).

Output per clip (--out_dir/<clip_id>/method1/):
  frames/frame_NNNNNN.png   — final composited frames
  metadata.json             — target_age, scale_factor, foot_anchor_rms_px

Usage
-----
  python 03a_pose_retarget.py \\
      --clips_json      selected_clips/selected_clips.json \\
      --masks_dir       pipeline_output/01_masks \\
      --erased_dir      pipeline_output/02_erased \\
      --video_dir       /data/videos \\
      --out_dir         pipeline_output/03a_method1 \\
      --target_age      child   # or adult or senior \\
      [--wan_checkpoint /path/to/Wan2.1-VACE-14B] \\
      [--height_ratio   0.60]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.bbox_ops import foot_anchor, foot_contact_rms
from utils.video_io import extract_frames, frames_to_array

# ---------------------------------------------------------------------------
# Height ratios relative to adult pedestrian height
# Derived from JAAD/IDD-PeD bounding-box statistics + CDC growth charts
# ---------------------------------------------------------------------------
HEIGHT_RATIOS = {
    "child":  0.60,
    "adult":  1.00,
    "senior": 0.95,
}

# Text prompts for Wan2.1-VACE inpainting (age identity conditioning)
AGE_PROMPTS = {
    "child":  ("a young child pedestrian, about 8 years old, walking along a road, "
               "realistic, high quality, consistent lighting"),
    "adult":  ("an adult pedestrian, walking along a road, "
               "realistic, high quality, consistent lighting"),
    "senior": ("an elderly pedestrian, about 70 years old, walking carefully along a road, "
               "realistic, high quality, consistent lighting"),
}


# ---------------------------------------------------------------------------
# Pose extraction (DWPose / OpenPose wrapper)
# ---------------------------------------------------------------------------

def extract_pose_sequence(
    frames: np.ndarray,         # (T, H, W, 3) uint8 RGB
    bboxes: Dict[int, List],    # frame_num → [xtl, ytl, xbr, ybr]
    start_frame: int,
) -> Optional[np.ndarray]:
    """
    Extract 2D keypoints for each frame using DWPose.
    Returns array of shape (T, 18, 2) [COCO 18-keypoint format] or None.

    Falls back to a simple linear interpolation skeleton if DWPose is
    not installed (for pipeline testing).
    """
    try:
        from dwpose import DWposeDetector  # type: ignore
        detector = DWposeDetector()

        all_kps = []
        for i, frame in enumerate(frames):
            fn = start_frame + i
            bbox = bboxes.get(str(fn), bboxes.get(fn))
            if bbox is None:
                all_kps.append(None)
                continue
            xtl, ytl, xbr, ybr = [int(v) for v in bbox]
            crop = frame[ytl:ybr, xtl:xbr]
            if crop.size == 0:
                all_kps.append(None)
                continue
            result = detector(crop)
            kps = result.get("bodies", {}).get("candidate")
            if kps is not None and len(kps) > 0:
                # Translate back to full-frame coords
                kps_full = kps.copy()
                kps_full[:, 0] += xtl
                kps_full[:, 1] += ytl
                all_kps.append(kps_full)
            else:
                all_kps.append(None)
        return all_kps

    except ImportError:
        raise RuntimeError("DWPose is required for pose-retargeting; no text-only fallback is valid.")


def scale_pose_sequence(
    kps_list: Optional[List],
    bboxes: Dict,
    start_frame: int,
    height_ratio: float,
    foot_anchors: Dict[str, List],
) -> Optional[List]:
    """
    Scale keypoints about the foot-anchor to simulate a different-height actor.

    Strategy:
      1. Identify the foot keypoint (ankle average, keypoint indices 15 & 16
         in COCO 18-keypoint format).
      2. Scale all keypoints by height_ratio relative to the foot anchor.

    Returns the scaled keypoints list, or None if input is None.
    """
    if kps_list is None:
        return None

    scaled = []
    for i, kps in enumerate(kps_list):
        fn_str = str(start_frame + i)
        fa = foot_anchors.get(fn_str)
        if kps is None or fa is None:
            scaled.append(kps)
            continue

        foot_x, foot_y = fa
        anchor = np.array([foot_x, foot_y], dtype=np.float32)

        # Scale keypoints about the foot anchor
        kps_arr = np.array(kps, dtype=np.float32)
        kps_arr = anchor + (kps_arr - anchor) * height_ratio

        scaled.append(kps_arr)

    return scaled


# ---------------------------------------------------------------------------
# Wan2.1-VACE interface
# ---------------------------------------------------------------------------

def run_wan_vace(
    erased_frames_dir: str,
    pose_frames_dir: Optional[str],
    output_dir: str,
    prompt: str,
    wan_checkpoint: str,
    bbox_sequence: List[List],
    target_age: str,
) -> bool:
    """
    Invoke Wan2.1-VACE for pose-conditioned video inpainting / generation.

    Missing Wan checkpoint or a failed generation is a hard error.
    Returns True on success.
    """
    wan_script = Path(wan_checkpoint) / "generate.py"
    if not wan_script.exists():
        raise FileNotFoundError(f"Wan checkpoint/script not found: {wan_script}")

    cmd = [
        "python", str(wan_script),
        "--task",           "i2v-14B",
        "--prompt",         prompt,
        "--image",          erased_frames_dir,
        "--output_dir",     output_dir,
        "--ckpt_dir",       wan_checkpoint,
        "--sample_steps",   "50",
        "--sample_guide_scale", "5.0",
    ]
    if pose_frames_dir:
        cmd += ["--control_video", pose_frames_dir]

    print(f"  Running Wan2.1-VACE: {' '.join(cmd[:8])} ...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Wan generation failed (exit {result.returncode}); see Wan output above.")
    return True


# ---------------------------------------------------------------------------
# Compositing: paste generated actor back onto erased plate
# ---------------------------------------------------------------------------

def composite_onto_plate(
    erased_frames_dir: str,
    generated_frames_dir: str,
    masks_dir: str,
    out_dir: str,
    dilation_px: int = 5,
) -> None:
    """
    For each frame, alpha-composite the generated pedestrian region
    onto the erased background plate using the SAM2 mask (dilated slightly
    to avoid hard edges).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1,) * 2)

    for p in sorted(Path(erased_frames_dir).glob("frame_*.png")):
        plate = cv2.imread(str(p))
        gen   = cv2.imread(str(Path(generated_frames_dir) / p.name))
        mask_p = Path(masks_dir) / p.name
        mask  = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) if mask_p.exists() else None

        if plate is None or gen is None:
            continue

        if mask is None:
            # No mask: use generated frame as-is
            cv2.imwrite(str(out_dir / p.name), gen)
            continue

        # Dilate mask for smooth blending
        dilated = cv2.dilate(mask, kernel)
        alpha   = dilated.astype(np.float32) / 255.0
        alpha   = alpha[:, :, np.newaxis]

        composited = (gen * alpha + plate * (1.0 - alpha)).astype(np.uint8)
        cv2.imwrite(str(out_dir / p.name), composited)


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------

def process_clip(
    clip: Dict,
    target_age: str,
    masks_dir: Path,
    erased_dir: Path,
    video_dir: str,
    out_dir: Path,
    wan_checkpoint: str,
) -> Dict:
    clip_id    = clip["clip_id"]
    start_f    = int(clip["start_frame"])
    source_age = clip["age_bracket"]

    clip_out = out_dir / clip_id / "method1"
    clip_out.mkdir(parents=True, exist_ok=True)

    # Load foot anchors from Step 1
    fa_path = masks_dir / clip_id / "foot_anchors.json"
    if not fa_path.exists():
        print(f"  [SKIP] foot_anchors.json not found for {clip_id}")
        return {}
    with open(fa_path) as f:
        foot_anchors = json.load(f)

    # Load bboxes from Step 1
    bb_path = masks_dir / clip_id / "bboxes.json"
    if not bb_path.exists():
        print(f"  [SKIP] bboxes.json not found for {clip_id}")
        return {}
    with open(bb_path) as f:
        bboxes = json.load(f)

    erased_frames_dir = erased_dir / clip_id / "erased_frames"
    if not erased_frames_dir.exists():
        print(f"  [SKIP] Erased frames not found for {clip_id}")
        return {}

    # Determine height ratio for the target age
    source_ratio = HEIGHT_RATIOS.get(source_age, 1.0)
    height_ratio = HEIGHT_RATIOS[target_age] / source_ratio
    prompt = AGE_PROMPTS[target_age]
    print(f"  Target age: {target_age}  height_ratio={height_ratio:.2f}")

    # Pose must come from the original actor. Extracting it from the erased
    # plate (the previous implementation) produces no source skeleton.
    pose_frames_dir = None
    video_path = _find_video(video_dir, clip.get("dataset", "IDD-PeD"), clip["video_id"])
    if video_path is None:
        raise FileNotFoundError(f"Original video missing for {clip_id}; pose cannot be recovered.")
    with tempfile.TemporaryDirectory(prefix="discretio_pose_") as raw_dir:
        n = extract_frames(video_path, raw_dir, start_frame=start_f, end_frame=int(clip["end_frame"]))
        if n == 0:
            raise RuntimeError(f"No source frames extracted for pose conditioning: {clip_id}")
        frames_arr = frames_to_array(raw_dir)
        kps_list = extract_pose_sequence(frames_arr, bboxes, start_f)
        kps_scaled = scale_pose_sequence(kps_list, bboxes, start_f, height_ratio, foot_anchors)

        if kps_scaled is not None:
            pose_dir = clip_out / "pose_frames"
            pose_dir.mkdir(exist_ok=True)
            _render_pose_frames(kps_scaled, frames_arr.shape[2], frames_arr.shape[1],
                                pose_dir, start_f)
            pose_frames_dir = str(pose_dir)

    # Run Wan2.1-VACE
    gen_dir = clip_out / "wan_output"
    gen_dir.mkdir(exist_ok=True)
    wan_ok = run_wan_vace(
        str(erased_frames_dir), pose_frames_dir,
        str(gen_dir), prompt, wan_checkpoint,
        list(bboxes.values()), target_age,
    )
    if not wan_ok:
        raise RuntimeError("Pose retarget generation did not complete; refusing placeholder evaluation.")

    # Composite onto erased plate
    final_dir = clip_out / "frames"
    composite_onto_plate(
        str(erased_frames_dir), str(gen_dir),
        str(masks_dir / clip_id / "masks"),
        str(final_dir),
    )

    # Compute foot-contact RMS (quality metric)
    rms = _compute_foot_rms(foot_anchors, final_dir, bboxes, start_f)

    metadata = {
        "clip_id":       clip_id,
        "source_age":    source_age,
        "target_age":    target_age,
        "source_height_prior": source_ratio,
        "height_ratio":  height_ratio,
        "wan_success":   wan_ok,
        "foot_contact_rms_px": rms,
    }
    with open(clip_out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ {clip_id} → Method 1 complete.  Foot-contact RMS = {rms:.1f} px")
    return metadata


def _render_pose_frames(
    kps_list: List,
    width: int, height: int,
    out_dir: Path,
    start_frame: int,
) -> None:
    """Render stick-figure pose frames for Wan2.1-VACE conditioning."""
    # COCO 18-keypoint skeleton connectivity
    SKELETON = [
        (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
        (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
        (0,14),(14,16),(0,15),(15,17),
    ]
    for i, kps in enumerate(kps_list):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if kps is not None:
            kps_arr = np.array(kps)
            for a, b in SKELETON:
                if a < len(kps_arr) and b < len(kps_arr):
                    pt_a = tuple(kps_arr[a].astype(int))
                    pt_b = tuple(kps_arr[b].astype(int))
                    cv2.line(canvas, pt_a, pt_b, (255, 255, 255), 2)
            for pt in kps_arr:
                cv2.circle(canvas, tuple(pt.astype(int)), 3, (0, 255, 0), -1)
        cv2.imwrite(str(out_dir / f"frame_{start_frame + i:06d}.png"), canvas)


def _compute_foot_rms(
    orig_anchors: Dict,
    final_dir: Path,
    bboxes: Dict,
    start_frame: int,
) -> float:
    """
    Approximate foot-contact RMS: compare original foot anchors to
    bottom-centre of the edited pedestrian bounding box.
    Uses bboxes from SAM2 (original) vs visual centroid of the final mask.
    Since we don't re-run SAM2 on the edited clip, we use the original
    foot anchor as reference and report how much the composite shifted it.
    """
    # Source anchors cannot measure edited anchors. This stays unavailable
    # until an independent person tracker is run on final_dir.
    return float("nan")


def _find_video(video_dir: str, dataset: str, video_id: str) -> Optional[str]:
    root = Path(video_dir)
    for base in (root, root / dataset):
        for ext in (".mp4", ".avi", ".mov", ".MP4"):
            direct = base / f"{video_id}{ext}"
            if direct.is_file():
                return str(direct)
            matches = sorted(base.glob(f"**/{video_id}{ext}"))
            if matches:
                return str(matches[0])
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Method 1: Erase + Pose-Retargeted Reinsertion (Wan2.1-VACE)."
    )
    parser.add_argument("--clips_json",     required=True)
    parser.add_argument("--masks_dir",      default="pipeline_output/01_masks")
    parser.add_argument("--erased_dir",     default="pipeline_output/02_erased")
    parser.add_argument("--video_dir",      required=True)
    parser.add_argument("--out_dir",        default="pipeline_output/03a_method1")
    parser.add_argument("--target_age",     choices=["child", "adult", "senior"],
                        required=True, help="Age bracket to synthesise.")
    parser.add_argument("--wan_checkpoint", default="Wan2.1-VACE-14B",
                        help="Path to Wan2.1-VACE model directory.")
    parser.add_argument("--clip_ids",       nargs="*", default=None)
    args = parser.parse_args()

    with open(args.clips_json) as f:
        clips = json.load(f)
    if args.clip_ids:
        clips = [c for c in clips if c["clip_id"] in args.clip_ids]

    out_dir   = Path(args.out_dir)
    masks_dir = Path(args.masks_dir)
    erased_dir = Path(args.erased_dir)

    all_meta = []
    for clip in clips:
        print(f"\n[Clip] {clip['clip_id']}  source_age={clip['age_bracket']}  "
              f"→  target_age={args.target_age}")
        meta = process_clip(
            clip, args.target_age, masks_dir, erased_dir,
            args.video_dir, out_dir, args.wan_checkpoint,
        )
        if meta:
            all_meta.append(meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"method1_{args.target_age}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\nMethod 1 summary written to: {summary_path}")


if __name__ == "__main__":
    main()
