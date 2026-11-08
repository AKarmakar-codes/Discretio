"""
utils/bbox_ops.py
-----------------
Bounding-box utilities for the DISCRETIO counterfactual pipeline.

Key concepts:
  - foot_anchor: the bottom-centre pixel of a bounding box; used as the
    "ground contact point" that must stay fixed across counterfactual edits.
  - expanded_bbox: a bbox grown by a scale factor around its centre, used
    for Method 2 (unified mask-box inpainting) to give the diffusion model
    enough context to synthesise a different-size pedestrian.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Type alias for a bounding box: (xtl, ytl, xbr, ybr) in pixel coordinates
BBox = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Foot anchor
# ---------------------------------------------------------------------------

def foot_anchor(bbox: BBox) -> Tuple[float, float]:
    """
    Returns the foot-contact point: (x_centre_bottom, y_bottom).
    This is the bottom-centre of the bounding box.

    For consistency across frames the foot-anchor is computed on the
    *original* (pre-edit) bounding boxes so that the metric is not
    influenced by the edited pedestrian's possibly different height.
    """
    xtl, ytl, xbr, ybr = bbox
    return ((xtl + xbr) / 2.0, ybr)


def foot_anchor_sequence(
    track_boxes: Dict[int, BBox],
    frames: List[int],
) -> np.ndarray:
    """
    Compute foot anchors for a list of frame indices.

    Parameters
    ----------
    track_boxes : dict mapping frame_id → BBox
    frames      : list of frame ids to include (in order)

    Returns
    -------
    anchors : float32 array of shape (len(frames), 2) → columns: (x, y)
    """
    anchors = []
    for f in frames:
        if f in track_boxes:
            anchors.append(foot_anchor(track_boxes[f]))
        else:
            anchors.append((np.nan, np.nan))
    return np.array(anchors, dtype=np.float32)


def foot_contact_rms(
    original_anchors: np.ndarray,
    edited_anchors: np.ndarray,
) -> float:
    """
    Compute RMS pixel distance between original and edited foot-anchor
    sequences, ignoring NaN rows.

    Parameters
    ----------
    original_anchors, edited_anchors : (T, 2) float32 arrays

    Returns
    -------
    rms_px : float — lower is better (target ≤ 8 px)
    """
    diff = original_anchors - edited_anchors
    valid = ~np.any(np.isnan(diff), axis=1)
    if valid.sum() == 0:
        return float("nan")
    sq = np.sum(diff[valid] ** 2, axis=1)
    return float(np.sqrt(sq.mean()))


# ---------------------------------------------------------------------------
# BBox expansion (Method 2: unified mask-box inpainting)
# ---------------------------------------------------------------------------

def expand_bbox(
    bbox: BBox,
    scale: float = 1.4,
    image_width: int = 1920,
    image_height: int = 1080,
    aspect_ratio: float | None = None,
) -> BBox:
    """
    Expand a bounding box by `scale` about its centre while preserving the
    image boundary.  An optional `aspect_ratio` (w/h) can be imposed to
    ensure the diffusion model's inpaint region has the right proportions
    for the target actor size (e.g. child is shorter → narrower box).

    Parameters
    ----------
    bbox         : original (xtl, ytl, xbr, ybr)
    scale        : linear expansion factor (default 1.4)
    image_width  : frame width in pixels (for clamping)
    image_height : frame height in pixels (for clamping)
    aspect_ratio : optional desired w/h of the expanded box

    Returns
    -------
    expanded BBox, clamped to image boundaries
    """
    xtl, ytl, xbr, ybr = bbox
    cx = (xtl + xbr) / 2.0
    cy = (ytl + ybr) / 2.0
    w  = (xbr - xtl) * scale
    h  = (ybr - ytl) * scale

    if aspect_ratio is not None:
        # Adjust width/height to match target aspect ratio,
        # keeping the larger dimension so we don't shrink the person.
        if w / h > aspect_ratio:
            h = w / aspect_ratio
        else:
            w = h * aspect_ratio

    new_xtl = max(0.0, cx - w / 2.0)
    new_ytl = max(0.0, cy - h / 2.0)
    new_xbr = min(float(image_width),  cx + w / 2.0)
    new_ybr = min(float(image_height), cy + h / 2.0)

    return (new_xtl, new_ytl, new_xbr, new_ybr)


def bbox_to_mask(
    bbox: BBox,
    image_width: int = 1920,
    image_height: int = 1080,
    value: int = 255,
) -> np.ndarray:
    """
    Convert a BBox to a binary uint8 mask of shape (H, W).
    Pixels inside the box are set to `value`; outside to 0.
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    xtl, ytl, xbr, ybr = bbox
    mask[int(ytl) : int(ybr), int(xtl) : int(xbr)] = value
    return mask


# ---------------------------------------------------------------------------
# Bounding-box from SAM2 mask (helper for 01_segment_pedestrian.py)
# ---------------------------------------------------------------------------

def mask_to_bbox(binary_mask: np.ndarray) -> BBox | None:
    """
    Compute the tight bounding box of a binary (H, W) mask.
    Returns None if the mask is empty.
    """
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any():
        return None
    ytl, ybr = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    xtl, xbr = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return (float(xtl), float(ytl), float(xbr), float(ybr))


# ---------------------------------------------------------------------------
# Actor-specific aspect ratios for Method 2
# ---------------------------------------------------------------------------

#: Empirical w/h priors derived from IDD-PeD bounding-box statistics.
#: Used to shape the expanded inpaint box so the diffusion model generates
#: an actor of the right height relative to width.
ACTOR_ASPECT_RATIOS: Dict[str, float] = {
    "child":  0.40,   # narrower, shorter
    "adult":  0.46,   # canonical adult
    "senior": 0.44,   # slightly narrower gait
}
