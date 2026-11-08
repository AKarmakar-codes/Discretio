#!/usr/bin/env python3
"""
05_metrics_auto.py
-------------------
Step 5 — Automated Perceptual & Kinematic Metrics.

Computes all three automated quality gates from the plan:

  Gate 1 — Background Preservation:
    PSNR ≥ 32 dB AND LPIPS ≤ 0.08 on pixels outside the union of original
    and edited masks.

  Gate 2 — Foot-Contact Consistency:
    RMS pixel distance between original foot anchor and edited pedestrian's
    foot anchor ≤ 8 px across t=-T..0 frames.

  Gate 3 — Detector Confidence:
    MiVOLO (or CLIP fallback) classifies the edited crops as the TARGET age
    bracket with ≥ 85% confidence.

Each gate is evaluated for BOTH methods (m1, m2) on all 10 benchmark clips.
Results feed directly into Table 6 and the empirical method selection.

Output
------
  metrics_report.csv        — per-clip per-method gate scores
  metrics_report.json       — same data as JSON
  method_comparison.txt     — which method wins per gate and overall

Usage
-----
  python 05_metrics_auto.py \\
      --clips_json      selected_clips/selected_clips.json \\
      --method1_dir     pipeline_output/03a_method1 \\
      --method2_dir     pipeline_output/03b_method2 \\
      --masks_dir       pipeline_output/01_masks \\
      --video_dir       /data/videos \\
      --out_dir         pipeline_output/05_metrics \\
      --target_age      child
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from utils.bbox_ops import foot_contact_rms
from utils.age_classifier import classify_age_sequence, CONFIDENCE_THRESHOLD

# ---------------------------------------------------------------------------
# Gate thresholds (from paper plan)
# ---------------------------------------------------------------------------
GATE1_MIN_PSNR_DB    = 32.0
GATE1_MAX_LPIPS      = 0.08
GATE2_MAX_FOOT_RMS   = 8.0    # pixels
GATE3_MIN_CONFIDENCE = CONFIDENCE_THRESHOLD  # 0.85


# ---------------------------------------------------------------------------
# Gate 1: Background Preservation (PSNR / LPIPS outside union mask)
# ---------------------------------------------------------------------------

def gate1_background(
    orig_frames_dir: str,
    edited_frames_dir: str,
    orig_masks_dir: str,
    edited_masks_dir: Optional[str],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (psnr_db, lpips) for the background region (outside union mask).
    Returns (None, None) if frames not found.
    """
    if edited_masks_dir is None or not Path(edited_masks_dir).exists():
        # Without an independently measured generated-actor mask, excluding
        # only the source box can score changed actor pixels as background.
        return None, None
    lpips_fn = None
    try:
        import lpips as lpips_lib
        import torch
        lpips_fn = lpips_lib.LPIPS(net="alex").eval()
    except ImportError:
        pass

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    psnrs, lpips_scores = [], []

    for p in sorted(Path(orig_frames_dir).glob("frame_*.png")):
        orig   = cv2.imread(str(p))
        edited = cv2.imread(str(Path(edited_frames_dir) / p.name))
        if orig is None or edited is None:
            continue

        orig_mask  = _load_mask(orig_masks_dir, p.name, target_shape=orig.shape[:2])
        edit_mask  = _load_mask(edited_masks_dir, p.name, target_shape=orig.shape[:2]) if edited_masks_dir else None

        # Union of original and edited masks
        union_mask = orig_mask.copy()
        if edit_mask is not None:
            union_mask = cv2.bitwise_or(union_mask, edit_mask)

        # Dilate to capture boundary pixels
        dilated = cv2.dilate(union_mask, kernel)
        outside = dilated == 0

        if outside.sum() < 500:
            continue

        # PSNR
        mse = np.mean((orig[outside].astype(np.float64) -
                       edited[outside].astype(np.float64)) ** 2)
        psnr = 100.0 if mse < 1e-6 else 10 * np.log10(255 ** 2 / mse)
        psnrs.append(psnr)

        # LPIPS on a background patch
        if lpips_fn is not None:
            try:
                import torch
                h, w = orig.shape[:2]
                ys, xs = np.where(outside)
                if len(ys) > 0:
                    idx = np.random.randint(len(ys))
                    y0 = max(0, ys[idx] - 128); y1 = min(h, y0 + 256)
                    x0 = max(0, xs[idx] - 128); x1 = min(w, x0 + 256)
                    if (y1 - y0) >= 64 and (x1 - x0) >= 64:
                        t_a = _to_lpips_tensor(orig[y0:y1, x0:x1])
                        t_b = _to_lpips_tensor(edited[y0:y1, x0:x1])
                        with torch.no_grad():
                            lpips_scores.append(float(lpips_fn(t_a, t_b).item()))
            except Exception:
                pass

    psnr = float(np.mean(psnrs))   if psnrs       else None
    lpips = float(np.mean(lpips_scores)) if lpips_scores else None
    return psnr, lpips


def _load_mask(masks_dir: Optional[str], fname: str, target_shape: Tuple[int, int] = (1080, 1920)) -> np.ndarray:
    if masks_dir:
        p = Path(masks_dir) / fname
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape[:2] != target_shape:
                    m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
                return m
    return np.zeros(target_shape, dtype=np.uint8)


def _to_lpips_tensor(bgr: np.ndarray):
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Gate 2: Foot-Contact Consistency
# ---------------------------------------------------------------------------

def gate2_foot_contact(
    orig_foot_anchors_path: str,
    edited_frames_dir: str,
    start_frame: int,
    end_frame: int,
) -> Optional[float]:
    """
    Compute foot-contact RMS between original anchors (from Step 1) and
    the foot anchor extracted from the edited frames.

    For the edited frames we use a simple background-subtraction heuristic
    or bounding box from the mask to find the new pedestrian's foot.

    Returns RMS in pixels, or None if data is unavailable.
    """
    # A valid edited foot anchor requires an independent detector/tracker on
    # the generated actor. Reusing the source annotation guarantees zero error
    # and is circular. Until such a tracker is configured, report unavailable.
    return None


# ---------------------------------------------------------------------------
# Gate 3: Detector Confidence
# ---------------------------------------------------------------------------

def gate3_detector(
    edited_frames_dir: str,
    orig_bboxes_path: str,
    target_age: str,
    max_frames: int = 20,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Run age classifier on edited pedestrian crops.

    Returns (classified_bracket, confidence) or (None, None).
    """
    if not Path(orig_bboxes_path).exists():
        return None, None
    with open(orig_bboxes_path) as f:
        bboxes = json.load(f)

    crops = []
    all_paths = sorted(Path(edited_frames_dir).glob("frame_*.png"))
    # Sample evenly across the clip
    step = max(1, len(all_paths) // max_frames)
    sampled = all_paths[::step][:max_frames]

    for p in sampled:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        fn = int(p.stem.split("_")[-1])
        bbox = bboxes.get(str(fn))
        if bbox is None:
            continue
        xtl, ytl, xbr, ybr = [int(v) for v in bbox]
        crop = frame[ytl:ybr, xtl:xbr]
        if crop.size > 0:
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    if not crops:
        return None, None

    result = classify_age_sequence(crops)
    return result.bracket, result.confidence


# ---------------------------------------------------------------------------
# Per-clip evaluation
# ---------------------------------------------------------------------------

def evaluate_clip(
    clip: Dict,
    target_age: str,
    method1_dir: Path,
    method2_dir: Path,
    masks_dir: Path,
    video_dir: str,
) -> Dict:
    clip_id  = clip["clip_id"]
    start_f  = int(clip["start_frame"])
    end_f    = int(clip["end_frame"])

    # Paths
    orig_masks_dir  = str(masks_dir / clip_id / "masks")
    orig_fa_path    = str(masks_dir / clip_id / "foot_anchors.json")
    orig_bboxes     = str(masks_dir / clip_id / "bboxes.json")

    # Locate raw (original) frames
    raw_frames_dir  = _find_or_extract_frames(video_dir, clip)

    result: Dict = {"clip_id": clip_id, "target_age": target_age,
                    "source_age": clip["age_bracket"]}

    for method_tag, method_dir in [("m1", method1_dir), ("m2", method2_dir)]:
        edited_dir_path = method_dir / clip_id / (
            "method1" if method_tag == "m1" else "method2"
        ) / "frames"

        if not edited_dir_path.exists():
            print(f"  [SKIP] {method_tag} frames not found for {clip_id}")
            for key in ["psnr", "lpips", "foot_rms", "age_pred", "age_conf",
                        "g1_pass", "g2_pass", "g3_pass", "all_pass"]:
                result[f"{method_tag}_{key}"] = None
            continue

        edited_str = str(edited_dir_path)
        print(f"  Evaluating {clip_id} [{method_tag}] ...")

        # Gate 1
        psnr, lpips = gate1_background(
            raw_frames_dir, edited_str, orig_masks_dir,
            edited_masks_dir=None,  # generated-actor masks are not yet available
        )
        g1 = (psnr is not None and psnr >= GATE1_MIN_PSNR_DB and
              lpips is not None and lpips <= GATE1_MAX_LPIPS)

        # Gate 2
        foot_rms = gate2_foot_contact(orig_fa_path, edited_str, start_f, end_f)
        g2 = foot_rms is not None and foot_rms <= GATE2_MAX_FOOT_RMS

        # Gate 3
        age_pred, age_conf = gate3_detector(edited_str, orig_bboxes, target_age)
        g3 = (age_pred == target_age and age_conf is not None and
              age_conf >= GATE3_MIN_CONFIDENCE)

        result[f"{method_tag}_psnr"]    = round(psnr,     2) if psnr     is not None else None
        result[f"{method_tag}_lpips"]   = round(lpips,    4) if lpips    is not None else None
        result[f"{method_tag}_foot_rms"]= round(foot_rms, 2) if foot_rms is not None else None
        result[f"{method_tag}_age_pred"]= age_pred
        result[f"{method_tag}_age_conf"]= round(age_conf, 4) if age_conf is not None else None
        result[f"{method_tag}_g1_pass"] = g1
        result[f"{method_tag}_g2_pass"] = g2
        result[f"{method_tag}_g3_pass"] = g3
        result[f"{method_tag}_all_pass"]= g1 and g2 and g3

        status = "✓ ALL" if (g1 and g2 and g3) else f"✗ G1={'✓' if g1 else '✗'} G2={'✓' if g2 else '✗'} G3={'✓' if g3 else '✗'}"
        psnr_s = f"{psnr:.1f}" if psnr is not None else "N/A"
        rms_s  = f"{foot_rms:.1f}" if foot_rms is not None else "N/A"
        conf_s = f"{age_conf:.2f}" if age_conf is not None else "N/A"
        print(f"    [{method_tag}] PSNR={psnr_s} dB  RMS={rms_s} px  "
              f"Age={age_pred}@{conf_s}  {status}")

    return result


def _find_or_extract_frames(video_dir: str, clip: Dict) -> str:
    """Return path to a directory of original extracted frames."""
    tmp = Path("/tmp") / f"discretio_orig_{clip['clip_id']}"
    if tmp.exists() and any(tmp.glob("frame_*.png")):
        return str(tmp)

    video_dir_p = Path(video_dir)
    for ext in (".mp4", ".avi", ".mov"):
        for root in [video_dir_p, video_dir_p / clip["dataset"]]:
            vp = root / f"{clip['video_id']}{ext}"
            if vp.exists():
                from utils.video_io import extract_frames
                tmp.mkdir(parents=True, exist_ok=True)
                extract_frames(str(vp), str(tmp),
                               start_frame=int(clip["start_frame"]),
                               end_frame=int(clip["end_frame"]))
                return str(tmp)
    return str(tmp)


# ---------------------------------------------------------------------------
# Method comparison report
# ---------------------------------------------------------------------------

def write_comparison(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "DISCRETIO — Automated Metrics: Method Comparison",
        "=" * 60,
        "",
        f"{'Clip':<45} {'G1_M1':>6} {'G1_M2':>6} {'G2_M1':>6} {'G2_M2':>6} "
        f"{'G3_M1':>6} {'G3_M2':>6} {'ALL_M1':>7} {'ALL_M2':>7}",
        "-" * 110,
    ]
    for _, row in df.iterrows():
        def fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "  N/A"
            return "  ✓ " if v else "  ✗ "
        lines.append(
            f"{row['clip_id']:<45} "
            f"{fmt(row.get('m1_g1_pass')):>6} {fmt(row.get('m2_g1_pass')):>6} "
            f"{fmt(row.get('m1_g2_pass')):>6} {fmt(row.get('m2_g2_pass')):>6} "
            f"{fmt(row.get('m1_g3_pass')):>6} {fmt(row.get('m2_g3_pass')):>6} "
            f"{fmt(row.get('m1_all_pass')):>7} {fmt(row.get('m2_all_pass')):>7}"
        )

    m1_wins = df["m1_all_pass"].fillna(False).sum() if "m1_all_pass" in df.columns else 0
    m2_wins = df["m2_all_pass"].fillna(False).sum() if "m2_all_pass" in df.columns else 0

    lines += [
        "-" * 110,
        f"Total clips passing ALL gates:  M1={m1_wins}  M2={m2_wins}",
        "",
        "No method is selected automatically. Review valid per-clip metrics, uncertainty, and blinded human ratings.",
        "",
        "Next step: Run 06_human_validation.py to confirm with Likert scores.",
    ]
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated metrics: PSNR/LPIPS, foot-contact RMS, age-classifier confidence."
    )
    parser.add_argument("--clips_json",   required=True)
    parser.add_argument("--method1_dir",  default="pipeline_output/03a_method1")
    parser.add_argument("--method2_dir",  default="pipeline_output/03b_method2")
    parser.add_argument("--masks_dir",    default="pipeline_output/01_masks")
    parser.add_argument("--video_dir",    required=True)
    parser.add_argument("--out_dir",      default="pipeline_output/05_metrics")
    parser.add_argument("--target_age",   choices=["child", "adult", "senior"],
                        default="adult")
    parser.add_argument("--clip_ids",     nargs="*", default=None)
    args = parser.parse_args()

    with open(args.clips_json) as f:
        clips = json.load(f)
    if args.clip_ids:
        clips = [c for c in clips if c["clip_id"] in args.clip_ids]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for clip in clips:
        print(f"\n[Clip] {clip['clip_id']}  target_age={args.target_age}")
        row = evaluate_clip(
            clip, args.target_age,
            Path(args.method1_dir), Path(args.method2_dir),
            Path(args.masks_dir), args.video_dir,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics_report.csv", index=False)
    df.to_json(out_dir / "metrics_report.json", orient="records", indent=2)
    write_comparison(df, out_dir / "method_comparison.txt")

    print(f"\nMetrics saved to: {out_dir}")
    print(f"See method_comparison.txt for empirical recommendation.")


if __name__ == "__main__":
    main()
