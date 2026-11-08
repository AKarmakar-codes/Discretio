#!/usr/bin/env python3
"""
01_segment_pedestrian.py
------------------------
Step 1 of the DISCRETIO counterfactual pipeline.

For each selected clip, runs SAM 2 (Segment Anything Model 2) to produce
per-frame binary masks of the target pedestrian.  The mask tracks the
pedestrian across the full clip window [start_frame, end_frame].

Outputs per clip (written to --out_dir/<clip_id>/):
  masks/frame_NNNNNN.png   — binary mask (255 = pedestrian, 0 = background)
  bboxes.json              — per-frame tight bounding box from the mask
  foot_anchors.json        — per-frame foot-anchor (x_centre, y_bottom)

SAM 2 prompt strategy
---------------------
  - Prompt: a single point click at the CENTRE of the annotated bbox at t0.
  - SAM 2 propagates the mask forward AND backward through the clip.
  - If mask area drops below MIN_MASK_AREA_PX2, we re-prompt at the
    nearest available annotated bbox centre.

Usage
-----
  python 01_segment_pedestrian.py \\
      --clips_json  selected_clips/selected_clips.json \\
      --video_dir   /data/videos \\
      --out_dir     ./pipeline_output/01_masks \\
      [--sam2_checkpoint sam2_hiera_large.pt] \\
      [--sam2_config     sam2_hiera_l.yaml]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.bbox_ops import mask_to_bbox, foot_anchor
from utils.video_io import extract_frames

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_MASK_AREA_PX2 = 500   # Re-prompt if mask shrinks below this (noise guard)
DEFAULT_SAM2_CHECKPOINT = "sam2_hiera_large.pt"
DEFAULT_SAM2_CONFIG     = "sam2_hiera_l.yaml"


# ---------------------------------------------------------------------------
# SAM 2 wrapper
# ---------------------------------------------------------------------------

def build_sam2_predictor(checkpoint: str, config: str):
    """
    Load SAM 2 VideoPredictor.  Returns None if sam2 is not installed
    (allowing dry-run / unit testing without the model).
    """
    try:
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        predictor = build_sam2_video_predictor(config, checkpoint)
        return predictor
    except ImportError:
        raise RuntimeError("SAM 2 is required; rectangular mock masks are not valid segmentation results.")


def mock_segment(
    frames: np.ndarray,
    prompt_bbox: Tuple[float, float, float, float],
) -> List[np.ndarray]:
    """
    Mock segmentation when SAM 2 is unavailable: fills the annotated bbox
    as a rectangular mask for all frames.  Used for pipeline testing.
    """
    H, W = frames.shape[1], frames.shape[2]
    xtl, ytl, xbr, ybr = prompt_bbox
    masks = []
    for _ in frames:
        m = np.zeros((H, W), dtype=np.uint8)
        m[int(ytl):int(ybr), int(xtl):int(xbr)] = 255
        masks.append(m)
    return masks


def run_sam2_on_clip(
    predictor,
    frames: np.ndarray,         # (T, H, W, 3) uint8 RGB
    prompt_bbox_t0: Tuple[float, float, float, float],
    t0_local_idx: int,          # index of t0 within the frames array
) -> List[np.ndarray]:
    """
    Run SAM 2 on a clip, prompted at t0 with the bbox centre.

    Returns a list of T binary masks (uint8, values 0 or 255).
    """
    if predictor is None:
        raise RuntimeError("SAM 2 predictor is unavailable; refusing mock segmentation.")

    import torch

    xtl, ytl, xbr, ybr = prompt_bbox_t0
    prompt_x = (xtl + xbr) / 2.0
    prompt_y = (ytl + ybr) / 2.0

    # SAM 2 video predictor expects frames as individual images in a directory.
    # We pass them via a temporary directory.
    import tempfile
    from utils.video_io import array_to_frames

    with tempfile.TemporaryDirectory() as tmpdir:
        array_to_frames(frames, tmpdir, start_index=0, rgb=True)

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device_type == "cuda" else __import__("contextlib").nullcontext()
        with torch.inference_mode(), autocast:
            inference_state = predictor.init_state(video_path=tmpdir)
            predictor.reset_state(inference_state)

            # Add positive prompt at t0
            _, _, mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=t0_local_idx,
                obj_id=1,
                points=[[prompt_x, prompt_y]],
                labels=[1],  # 1 = foreground
            )

            # Propagate forward then backward
            masks: Dict[int, np.ndarray] = {}
            for frame_idx, obj_ids, seg_masks in predictor.propagate_in_video(inference_state):
                m = (seg_masks[0].cpu().numpy() > 0.0).astype(np.uint8) * 255
                masks[frame_idx] = m.squeeze()

            for frame_idx, obj_ids, seg_masks in predictor.propagate_in_video(
                inference_state, reverse=True
            ):
                if frame_idx not in masks:
                    m = (seg_masks[0].cpu().numpy() > 0.0).astype(np.uint8) * 255
                    masks[frame_idx] = m.squeeze()

    return [masks.get(i, np.zeros(frames.shape[1:3], dtype=np.uint8)) for i in range(len(frames))]


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------

def process_clip(
    clip: Dict,
    video_dir: str,
    out_dir: Path,
    predictor,
) -> None:
    clip_id   = clip["clip_id"]
    dataset   = clip["dataset"]
    video_id  = clip["video_id"]
    start_f   = int(clip["start_frame"])
    end_f     = int(clip["end_frame"])
    t0_f      = int(clip["t0_frame"])
    bbox_t0   = (
        float(clip["xtl_t0"]), float(clip["ytl_t0"]),
        float(clip["xbr_t0"]), float(clip["ybr_t0"]),
    )

    clip_out = out_dir / clip_id
    clip_out.mkdir(parents=True, exist_ok=True)
    masks_dir = clip_out / "masks"
    masks_dir.mkdir(exist_ok=True)

    # --- Locate video file ---
    video_path = _find_video(video_dir, dataset, video_id)
    if video_path is None:
        print(f"  [SKIP] Video not found for clip {clip_id}")
        return

    print(f"  Extracting frames [{start_f}–{end_f}] from {os.path.basename(video_path)} ...")
    import tempfile
    with tempfile.TemporaryDirectory() as frame_tmp:
        n = extract_frames(video_path, frame_tmp, start_frame=start_f, end_frame=end_f)
        if n == 0:
            print(f"  [SKIP] No frames extracted for {clip_id}")
            return

        from utils.video_io import frames_to_array
        try:
            frames = frames_to_array(frame_tmp)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            return

    t0_local = t0_f - start_f   # index within the extracted array
    if not 0 <= t0_local < len(frames):
        raise ValueError(f"t0_frame {t0_f} is outside extracted clip [{start_f}, {start_f + len(frames) - 1}]")

    print(f"  Running SAM 2 (T={len(frames)} frames, t0_local={t0_local}) ...")
    raw_masks = run_sam2_on_clip(predictor, frames, bbox_t0, t0_local)

    # --- Re-prompt guard ---
    # If mask area at t0 is below threshold, the click landed outside the person.
    t0_mask_area = int(raw_masks[t0_local].sum()) // 255
    if t0_mask_area < MIN_MASK_AREA_PX2:
        print(f"  [WARN] Mask area at t0 is {t0_mask_area} px² < {MIN_MASK_AREA_PX2}. "
              f"Check prompt bbox for {clip_id}.")

    # --- Save masks and derive bboxes / foot anchors ---
    bboxes: Dict[str, list] = {}
    foot_anchors_out: Dict[str, list] = {}

    for i, mask in enumerate(raw_masks):
        frame_num = start_f + i
        fname = f"frame_{frame_num:06d}.png"
        cv2.imwrite(str(masks_dir / fname), mask)

        bbox = mask_to_bbox(mask)
        if bbox is not None:
            bboxes[str(frame_num)] = list(bbox)
            foot_anchors_out[str(frame_num)] = list(foot_anchor(bbox))

    with open(clip_out / "bboxes.json", "w") as f:
        json.dump(bboxes, f, indent=2)
    with open(clip_out / "foot_anchors.json", "w") as f:
        json.dump(foot_anchors_out, f, indent=2)

    print(f"  ✓ {clip_id}: {len(bboxes)} frames masked.")


def _find_video(video_dir: str, dataset: str, video_id: str) -> Optional[str]:
    """
    Search for a video file in video_dir matching video_id.
    Looks for .mp4, .avi, .mov extensions, optionally under dataset subfolder.
    """
    video_dir = Path(video_dir)
    for ext in (".mp4", ".avi", ".mov", ".MP4"):
        for search_root in [video_dir, video_dir / dataset]:
            candidate = search_root / f"{video_id}{ext}"
            if candidate.is_file():
                return str(candidate)
            # recursive search one level
            for p in search_root.glob(f"**/{video_id}{ext}"):
                return str(p)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Segment target pedestrians with SAM 2 for DISCRETIO counterfactual pipeline."
    )
    parser.add_argument("--clips_json",       required=True)
    parser.add_argument("--video_dir",        required=True,
                        help="Root directory containing raw videos.")
    parser.add_argument("--out_dir",          default="pipeline_output/01_masks")
    parser.add_argument("--sam2_checkpoint",  default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--sam2_config",      default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--clip_ids",         nargs="*", default=None,
                        help="Process only these clip_ids (default: all).")
    args = parser.parse_args()

    with open(args.clips_json) as f:
        clips = json.load(f)

    if args.clip_ids:
        clips = [c for c in clips if c["clip_id"] in args.clip_ids]
    print(f"Processing {len(clips)} clip(s).")

    predictor = build_sam2_predictor(args.sam2_checkpoint, args.sam2_config)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        print(f"\n[Clip] {clip['clip_id']}  age={clip['age_bracket']}  role={clip['role']}")
        process_clip(clip, args.video_dir, out_dir, predictor)

    print("\nDone. Masks written to:", out_dir)


if __name__ == "__main__":
    main()
