"""
utils/age_classifier.py
-----------------------
Age-bracket classifier wrapper for the DISCRETIO counterfactual pipeline.

Primary backend: MiVOLO (face + body crop → continuous age prediction).
Fallback:        CLIP zero-shot text probing when no face is detected.

Output is always one of: "child", "adult", "senior"
Confidence threshold: target ≥ 0.85 (paper §Step 2 Detector Confidence).

Age bins
--------
  child  : 0–12
  adult  : 13–59
  senior : 60+

The bins are deliberately coarse to match JAAD / IDD-PeD annotation scheme.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

AGE_BRACKETS = ("child", "adult", "senior")

# Age bin boundaries (inclusive lower, exclusive upper)
AGE_BINS: Dict[str, Tuple[int, int]] = {
    "child":  (0,  13),
    "adult":  (13, 60),
    "senior": (60, 120),
}

CONFIDENCE_THRESHOLD = 0.85


@dataclass
class AgeClassificationResult:
    bracket: str           # "child" | "adult" | "senior"
    confidence: float      # 0–1
    raw_age_estimate: Optional[float]  # continuous age from MiVOLO, or None
    method: str            # "mivolo" | "clip" | "fallback"
    passed_threshold: bool # confidence >= CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Age → bracket mapping
# ---------------------------------------------------------------------------

def age_to_bracket(age: float) -> str:
    """Map a continuous age estimate to a bracket string."""
    for bracket, (lo, hi) in AGE_BINS.items():
        if lo <= age < hi:
            return bracket
    return "unknown"


# ---------------------------------------------------------------------------
# MiVOLO backend
# ---------------------------------------------------------------------------

def _classify_mivolo(
    image_rgb: np.ndarray,  # (H, W, 3) uint8
    body_bbox: Optional[Tuple[int, int, int, int]] = None,  # (x1, y1, x2, y2)
) -> Optional[AgeClassificationResult]:
    """
    Run MiVOLO age estimation on an RGB crop.

    MiVOLO predicts a continuous age from face+body features. We convert
    the continuous prediction to a bracket and derive a soft confidence
    from the distance to the nearest bin boundary.

    Returns None if MiVOLO is not installed.
    """
    try:
        from mivolo.predictor import Predictor as MiVOLOPredictor  # type: ignore
    except ImportError:
        logger.debug("MiVOLO not installed; skipping.")
        return None

    try:
        predictor = MiVOLOPredictor()  # uses default checkpoint
        result = predictor.predict(image_rgb, body_bbox=body_bbox)
        raw_age = float(result["age"])
        bracket = age_to_bracket(raw_age)

        # Confidence: sigmoid of how far age is from the nearest boundary
        # centre of the bin gives confidence=1; boundary gives confidence=0.5
        lo, hi = AGE_BINS[bracket]
        centre  = (lo + hi) / 2.0
        half_w  = (hi - lo) / 2.0
        dist    = abs(raw_age - centre)
        # Normalised distance: 0 at centre → 1 at boundary
        norm_d  = min(dist / half_w, 1.0) if half_w > 0 else 0.0
        confidence = 1.0 - 0.5 * norm_d  # range: [0.5, 1.0]

        return AgeClassificationResult(
            bracket=bracket,
            confidence=confidence,
            raw_age_estimate=raw_age,
            method="mivolo",
            passed_threshold=confidence >= CONFIDENCE_THRESHOLD,
        )

    except Exception as e:
        logger.warning(f"MiVOLO inference failed: {e}")
        return None


# ---------------------------------------------------------------------------
# CLIP zero-shot fallback
# ---------------------------------------------------------------------------

_CLIP_PROMPTS: Dict[str, str] = {
    "child":  "a photo of a young child pedestrian",
    "adult":  "a photo of an adult pedestrian",
    "senior": "a photo of an elderly pedestrian",
}


def _classify_clip(image_rgb: np.ndarray) -> Optional[AgeClassificationResult]:
    """
    Zero-shot CLIP text-image similarity for age bracket classification.

    Uses cosine similarity between the image embedding and each bracket prompt.
    Returns None if CLIP is not installed.
    """
    try:
        import clip  # type: ignore
        import torch
        from PIL import Image
    except ImportError:
        logger.debug("CLIP not installed; skipping.")
        return None

    try:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        pil_img = Image.fromarray(image_rgb)
        image_input = preprocess(pil_img).unsqueeze(0).to(device)

        texts = list(_CLIP_PROMPTS.values())
        text_inputs = clip.tokenize(texts).to(device)

        with __import__("torch").no_grad():
            image_features = model.encode_image(image_input)
            text_features  = model.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features  /= text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(0)
            probs = sims.softmax(dim=-1).cpu().numpy()

        idx = int(np.argmax(probs))
        bracket = list(_CLIP_PROMPTS.keys())[idx]
        confidence = float(probs[idx])

        return AgeClassificationResult(
            bracket=bracket,
            confidence=confidence,
            raw_age_estimate=None,
            method="clip",
            passed_threshold=confidence >= CONFIDENCE_THRESHOLD,
        )

    except Exception as e:
        logger.warning(f"CLIP inference failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify_age(
    image_rgb: np.ndarray,
    body_bbox: Optional[Tuple[int, int, int, int]] = None,
    force_method: Optional[str] = None,
) -> AgeClassificationResult:
    """
    Classify the age bracket of a pedestrian crop.

    Strategy:
      1. MiVOLO (face+body, continuous age) — most accurate
      2. CLIP zero-shot fallback — when MiVOLO is unavailable / no face
      3. Unavailable fallback → "unknown" with confidence 0.0 (manual review)

    Parameters
    ----------
    image_rgb  : (H, W, 3) uint8 RGB crop of the pedestrian
    body_bbox  : optional (x1, y1, x2, y2) tight body bbox within the crop
    force_method : "mivolo" | "clip" | None (auto)

    Returns
    -------
    AgeClassificationResult
    """
    if force_method != "clip":
        result = _classify_mivolo(image_rgb, body_bbox)
        if result is not None:
            return result

    if force_method != "mivolo":
        result = _classify_clip(image_rgb)
        if result is not None:
            return result

    logger.error("Both MiVOLO and CLIP unavailable — returning fallback result.")
    return AgeClassificationResult(
        bracket="unknown",
        confidence=0.0,
        raw_age_estimate=None,
        method="fallback",
        passed_threshold=False,
    )


def classify_age_sequence(
    crops: list[np.ndarray],
    body_bboxes: Optional[list[Optional[Tuple[int, int, int, int]]]] = None,
    aggregate: str = "majority",
) -> AgeClassificationResult:
    """
    Classify age bracket over a sequence of pedestrian crops (e.g. multiple frames).

    Parameters
    ----------
    crops       : list of (H, W, 3) uint8 RGB crops
    body_bboxes : optional list of body bboxes (same length as crops)
    aggregate   : "majority" (most common bracket) or "mean_conf" (highest mean conf)

    Returns
    -------
    AgeClassificationResult with aggregated bracket and mean confidence
    """
    if body_bboxes is None:
        body_bboxes = [None] * len(crops)

    results = [classify_age(crop, bbox) for crop, bbox in zip(crops, body_bboxes)]
    if not results:
        return AgeClassificationResult(
            bracket="unknown", confidence=0.0, raw_age_estimate=None,
            method="unavailable", passed_threshold=False,
        )

    # Majority vote on bracket
    from collections import Counter
    counts = Counter(r.bracket for r in results)
    bracket = counts.most_common(1)[0][0]

    # Average confidence for the winning bracket
    conf = float(np.mean([r.confidence for r in results if r.bracket == bracket]))

    return AgeClassificationResult(
        bracket=bracket,
        confidence=conf,
        raw_age_estimate=None,
        method=results[0].method,
        passed_threshold=conf >= CONFIDENCE_THRESHOLD,
    )
