#!/usr/bin/env python3
"""
visualize_comparison.py
-----------------------
Visualization utility for the DISCRETIO counterfactual pipeline.

Allows inspecting both INPUTS and OUTPUTS side-by-side:
  1. Input: Original frame + annotated BBox + foot anchor point
  2. Intermediate 1: SAM 2 segmentation mask overlaid in translucent red
  3. Intermediate 2: ProPainter erased background plate
  4. Output 1: Method 1 (Erase + Retargeted Pose Insertion)
  5. Output 2: Method 2 (Unified Expanded-BBox Inpainting)

Can be used:
  - From CLI: generates side-by-side MP4 video, animated GIF, and PNG contact sheets.
  - Inside Kaggle / Jupyter: displays interactive videos and image grids in notebook cells.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core Image Compositing Helpers
# ---------------------------------------------------------------------------

def draw_header(image: np.ndarray, text: str, bg_color: Tuple[int, int, int] = (30, 30, 30)) -> np.ndarray:
    """Draw a dark title banner above an image."""
    h, w = image.shape[:2]
    banner_h = 40
    banner = np.full((banner_h, w, 3), bg_color, dtype=np.uint8)
    cv2.putText(
        banner, text, (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )
    return np.vstack([banner, image])


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), alpha: float = 0.5) -> np.ndarray:
    """Overlay a binary mask onto an image with color tint."""
    if mask is None:
        return image.copy()
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = image.copy()
    fg = mask > 128
    colored = np.zeros_like(image)
    colored[:] = color
    out[fg] = cv2.addWeighted(image[fg], 1.0 - alpha, colored[fg], alpha, 0)
    # Draw contour outline for crisp visualization
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out


def draw_pedestrian_annotations(image: np.ndarray, bbox: Optional[List[float]], foot_point: Optional[List[float]]) -> np.ndarray:
    """Draw bounding box and foot anchor point on the frame."""
    out = image.copy()
    if bbox is not None and len(bbox) == 4:
        xtl, ytl, xbr, ybr = [int(v) for v in bbox]
        cv2.rectangle(out, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
    if foot_point is not None and len(foot_point) == 2:
        fx, fy = int(foot_point[0]), int(foot_point[1])
        cv2.circle(out, (fx, fy), 6, (0, 0, 255), -1)  # Red dot for foot-ground anchor
        cv2.circle(out, (fx, fy), 8, (255, 255, 255), 2)
    return out


# ---------------------------------------------------------------------------
# Build Multi-Panel Frame
# ---------------------------------------------------------------------------

def build_comparison_frame(
    orig_bgr: np.ndarray,
    mask_gray: Optional[np.ndarray],
    erased_bgr: Optional[np.ndarray],
    m1_bgr: Optional[np.ndarray],
    m2_bgr: Optional[np.ndarray],
    bbox: Optional[List[float]] = None,
    foot_anchor: Optional[List[float]] = None,
    target_width_per_panel: int = 480,
    zoom_crop: bool = False,
) -> np.ndarray:
    """
    Combines up to 5 panels horizontally:
    [1. Input (Bbox+Foot)] | [2. SAM 2 Mask] | [3. Erased Plate] | [4. Method 1] | [5. Method 2]
    """
    h_orig, w_orig = orig_bgr.shape[:2]

    # If zoom_crop requested, crop tightly around pedestrian + margin
    if zoom_crop and bbox is not None:
        xtl, ytl, xbr, ybr = bbox
        cw, ch = (xbr - xtl), (ybr - ytl)
        pad_x, pad_y = int(cw * 0.8), int(ch * 0.4)
        x0 = max(0, int(xtl - pad_x))
        x1 = min(w_orig, int(xbr + pad_x))
        y0 = max(0, int(ytl - pad_y))
        y1 = min(h_orig, int(ybr + pad_y))
    else:
        x0, y0, x1, y1 = 0, 0, w_orig, h_orig

    # Panel 1: Original with annotations
    p1 = draw_pedestrian_annotations(orig_bgr[y0:y1, x0:x1], bbox, foot_anchor)
    p1 = draw_header(p1, "1. Input Video (Anchor)")

    # Panel 2: Mask overlay
    if mask_gray is not None:
        p2 = overlay_mask(orig_bgr[y0:y1, x0:x1], mask_gray[y0:y1, x0:x1])
    else:
        p2 = np.zeros_like(p1)
    p2 = draw_header(p2, "2. SAM 2 Segmentation")

    # Panel 3: Erased background plate
    if erased_bgr is not None:
        p3 = erased_bgr[y0:y1, x0:x1].copy()
    else:
        p3 = np.zeros_like(p1)
    p3 = draw_header(p3, "3. Erased Plate")

    # Panel 4: Method 1 (Pose retargeted)
    if m1_bgr is not None:
        p4 = m1_bgr[y0:y1, x0:x1].copy()
    else:
        p4 = np.zeros_like(p1)
    p4 = draw_header(p4, "4. Method 1 (Pose Retarget)")

    # Panel 5: Method 2 (Unified Inpaint)
    if m2_bgr is not None:
        p5 = m2_bgr[y0:y1, x0:x1].copy()
    else:
        p5 = np.zeros_like(p1)
    p5 = draw_header(p5, "5. Method 2 (Unified Inpaint)")

    panels = [p1, p2, p3, p4, p5]

    # Resize all panels to target width while keeping aspect ratio
    resized_panels = []
    for p in panels:
        ph, pw = p.shape[:2]
        new_h = int(ph * (target_width_per_panel / max(1, pw)))
        resized = cv2.resize(p, (target_width_per_panel, new_h), interpolation=cv2.INTER_AREA)
        resized_panels.append(resized)

    # Ensure uniform height across all panels
    max_h = max(p.shape[0] for p in resized_panels)
    padded = []
    for p in resized_panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    return np.hstack(padded)


# ---------------------------------------------------------------------------
# Video and GIF Exporters
# ---------------------------------------------------------------------------

def export_clip_comparison(
    clip_id: str,
    pipeline_output_dir: str,
    out_dir: str,
    fps: float = 15.0,
    zoom_crop: bool = False,
    max_frames: int = 60,
    target_width_per_panel: int = 360,
) -> Tuple[str, str]:
    """
    Renders an entire clip comparison as MP4 and animated GIF.
    Returns (mp4_path, gif_path).
    """
    base = Path(pipeline_output_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # File paths - check both flat and nested method dirs
    orig_candidates = [
        base / "03b_method2" / clip_id / "raw_frames",
        base / "03b_method2" / clip_id / "method2" / "raw_frames",
        base / "03a_method1" / clip_id / "raw_frames",
        base / "03a_method1" / clip_id / "method1" / "raw_frames",
        base / "01_masks" / clip_id / "raw_frames",
    ]
    orig_dir = next((c for c in orig_candidates if c.exists() and list(c.glob("frame_*.png"))), None)
    mask_dir   = base / "01_masks" / clip_id / "masks"
    erased_dir = base / "02_erased" / clip_id / "erased_frames"
    
    m1_candidates = [
        base / "03a_method1" / clip_id / "frames",
        base / "03a_method1" / clip_id / "method1" / "frames",
    ]
    m1_dir = next((c for c in m1_candidates if c.exists()), base / "03a_method1" / clip_id / "method1" / "frames")

    m2_candidates = [
        base / "03b_method2" / clip_id / "frames",
        base / "03b_method2" / clip_id / "method2" / "frames",
    ]
    m2_dir = next((c for c in m2_candidates if c.exists()), base / "03b_method2" / clip_id / "method2" / "frames")

    # Metadata
    bbox_path = base / "01_masks" / clip_id / "bboxes.json"
    fa_path   = base / "01_masks" / clip_id / "foot_anchors.json"

    bboxes = json.load(open(bbox_path)) if bbox_path.exists() else {}
    foot_anchors = json.load(open(fa_path)) if fa_path.exists() else {}

    # Gather frame files
    if orig_dir is None:
        raise FileNotFoundError(f"No raw frames found for {clip_id} in any searched location.")
    orig_files = sorted(orig_dir.glob("frame_*.png"))

    step = max(1, len(orig_files) // max_frames)
    sampled_files = orig_files[::step][:max_frames]

    frames_out = []
    for fp in sampled_files:
        fn_str = str(int(fp.stem.split("_")[-1]))
        orig   = cv2.imread(str(fp))
        if orig is None:
            continue

        mask_p = mask_dir / fp.name
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) if mask_p.exists() else None

        erased_p = erased_dir / fp.name
        erased = cv2.imread(str(erased_p)) if erased_p.exists() else None

        m1_p = m1_dir / fp.name
        m1 = cv2.imread(str(m1_p)) if m1_p.exists() else None

        m2_p = m2_dir / fp.name
        m2 = cv2.imread(str(m2_p)) if m2_p.exists() else None

        composite = build_comparison_frame(
            orig_bgr=orig,
            mask_gray=mask,
            erased_bgr=erased,
            m1_bgr=m1,
            m2_bgr=m2,
            bbox=bboxes.get(fn_str),
            foot_anchor=foot_anchors.get(fn_str),
            target_width_per_panel=target_width_per_panel,
            zoom_crop=zoom_crop,
        )
        frames_out.append(composite)

    if not frames_out:
        raise ValueError(f"No composite frames could be rendered for {clip_id}")

    prefix = f"{clip_id}_zoom" if zoom_crop else f"{clip_id}_full"
    mp4_path = str(out_p / f"{prefix}_comparison.mp4")
    gif_path = str(out_p / f"{prefix}_comparison.gif")

    # 1. Write MP4
    h, w = frames_out[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
    for f in frames_out:
        out_video.write(f)
    out_video.release()

    # 2. Write GIF for immediate in-notebook display (PIL)
    from PIL import Image
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_out]
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=int(1000 / fps), loop=0
    )

    return mp4_path, gif_path


# ---------------------------------------------------------------------------
# Keyframe Grid Plot (Static Image)
# ---------------------------------------------------------------------------

def export_keyframe_grid(
    clip_id: str,
    pipeline_output_dir: str,
    out_dir: str,
    n_frames: int = 3,
    zoom_crop: bool = True,
) -> str:
    """Exports a static comparison image with n_frames keyframes down the timeline."""
    base = Path(pipeline_output_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    orig_candidates = [
        base / "03b_method2" / clip_id / "raw_frames",
        base / "03b_method2" / clip_id / "method2" / "raw_frames",
        base / "03a_method1" / clip_id / "raw_frames",
        base / "03a_method1" / clip_id / "method1" / "raw_frames",
        base / "01_masks" / clip_id / "raw_frames",
    ]
    orig_dir = next((c for c in orig_candidates if c.exists() and list(c.glob("frame_*.png"))), None)
    mask_dir = base / "01_masks" / clip_id / "masks"
    erased_dir = base / "02_erased" / clip_id / "erased_frames"
    
    m1_candidates = [
        base / "03a_method1" / clip_id / "frames",
        base / "03a_method1" / clip_id / "method1" / "frames",
    ]
    m1_dir = next((c for c in m1_candidates if c.exists()), base / "03a_method1" / clip_id / "method1" / "frames")

    m2_candidates = [
        base / "03b_method2" / clip_id / "frames",
        base / "03b_method2" / clip_id / "method2" / "frames",
    ]
    m2_dir = next((c for c in m2_candidates if c.exists()), base / "03b_method2" / clip_id / "method2" / "frames")

    bbox_path = base / "01_masks" / clip_id / "bboxes.json"
    fa_path = base / "01_masks" / clip_id / "foot_anchors.json"

    bboxes = json.load(open(bbox_path)) if bbox_path.exists() else {}
    foot_anchors = json.load(open(fa_path)) if fa_path.exists() else {}

    if orig_dir is None:
        raise FileNotFoundError(f"No raw frames found for {clip_id} in any searched location.")
    orig_files = sorted(orig_dir.glob("frame_*.png"))

    # Select early, middle (t0), and late frames
    indices = np.linspace(0, len(orig_files) - 1, n_frames, dtype=int)
    row_images = []

    for idx in indices:
        fp = orig_files[idx]
        fn_str = str(int(fp.stem.split("_")[-1]))
        orig = cv2.imread(str(fp))

        mask = cv2.imread(str(mask_dir / fp.name), cv2.IMREAD_GRAYSCALE) if (mask_dir / fp.name).exists() else None
        erased = cv2.imread(str(erased_dir / fp.name)) if (erased_dir / fp.name).exists() else None
        m1 = cv2.imread(str(m1_dir / fp.name)) if (m1_dir / fp.name).exists() else None
        m2 = cv2.imread(str(m2_dir / fp.name)) if (m2_dir / fp.name).exists() else None

        composite = build_comparison_frame(
            orig_bgr=orig,
            mask_gray=mask,
            erased_bgr=erased,
            m1_bgr=m1,
            m2_bgr=m2,
            bbox=bboxes.get(fn_str),
            foot_anchor=foot_anchors.get(fn_str),
            target_width_per_panel=320,
            zoom_crop=zoom_crop,
        )
        # Add frame timestamp stamp
        composite = draw_header(composite, f"Timeline Step {idx+1}/{n_frames} (Frame #{fn_str})", bg_color=(50, 20, 20))
        row_images.append(composite)

    full_grid = np.vstack(row_images)
    grid_path = str(out_p / f"{clip_id}_keyframe_grid.png")
    cv2.imwrite(grid_path, full_grid)
    return grid_path


# ---------------------------------------------------------------------------
# Notebook Interactive Helper
# ---------------------------------------------------------------------------

def display_clip_in_notebook(clip_id: str, pipeline_output_dir: str = "pipeline_output", zoom_crop: bool = True):
    """Call this inside a Kaggle / Jupyter cell to directly view the animation & grid."""
    from IPython.display import Image, display

    print(f"Generating visual comparison for: {clip_id} (zoom={zoom_crop}) ...")
    out_dir = f"{pipeline_output_dir}/visualizations"
    _, gif_path = export_clip_comparison(
        clip_id, pipeline_output_dir, out_dir=out_dir, zoom_crop=zoom_crop, fps=12, max_frames=45
    )
    grid_path = export_keyframe_grid(clip_id, pipeline_output_dir, out_dir=out_dir, n_frames=3, zoom_crop=zoom_crop)

    print("\n1. Keyframe Progression Grid (Static):")
    display(Image(filename=grid_path))

    print("\n2. Side-by-Side Video Animation (Inputs & Outputs):")
    display(Image(filename=gif_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render comparison of Inputs vs Outputs.")
    parser.add_argument("--clip_id", required=True, help="Clip ID to visualize.")
    parser.add_argument("--pipeline_output_dir", default="pipeline_output", help="Root pipeline output folder.")
    parser.add_argument("--out_dir", default="pipeline_output/visualizations", help="Output directory.")
    parser.add_argument("--zoom_crop", action="store_true", help="Crop around pedestrian instead of wide shot.")
    parser.add_argument("--max_frames", type=int, default=60, help="Max frames to include in video/GIF.")
    parser.add_argument("--fps", type=float, default=15.0, help="Frame rate.")
    args = parser.parse_args()

    mp4_p, gif_p = export_clip_comparison(
        args.clip_id, args.pipeline_output_dir, args.out_dir,
        fps=args.fps, zoom_crop=args.zoom_crop, max_frames=args.max_frames
    )
    grid_p = export_keyframe_grid(args.clip_id, args.pipeline_output_dir, args.out_dir, zoom_crop=args.zoom_crop)

    print("\nVisualizations successfully saved:")
    print(f"  MP4 Video : {mp4_p}")
    print(f"  GIF Anim  : {gif_p}")
    print(f"  Static PNG: {grid_p}")


if __name__ == "__main__":
    main()
