#!/usr/bin/env python3
"""
04_sanity_self_replace.py
--------------------------
Step 4 — The "Self-Replacement" Sanity Check (Artifact Isolation Test).

Formally tests the core claim for A* reviewers:
  The rollout divergence D(M(O_real), M(O_recon)) must be near zero.

This confirms that any divergence seen in the adult→child swap is due to
ACTOR IDENTITY (the behavioral prior we're measuring), NOT diffusion
inpainting noise.

Experiment
----------
  For each clip tagged as role="sanity_check" (adult-only clips):
    1. Take the ORIGINAL adult clip O_real.
    2. Run BOTH pipeline methods to re-insert an ADULT (same age bracket).
       This produces O_recon_m1 and O_recon_m2.
    3. Feed O_real and each O_recon to the world model (V-JEPA 2 / Cosmos).
    4. Compute rollout divergence D(M(O_real), M(O_recon)).
    5. Report: D_self ≪ D_swap confirms pipeline validity.

Rollout Divergence
------------------
  D is computed as the DISCRETIO metric (§4.1):
    D = λτ·Dτ + λo·Do + λu·Du + λi·Di + λt·Dt
  where each component is the Jensen–Shannon divergence of the corresponding
  rollout distribution.

  For this sanity check we use a simplified scalar proxy:
    D_traj = mean L2 distance between trajectory rollout means (pixels).

Output
------
  sanity_check_results.json
    Per clip: {clip_id, D_self_m1, D_self_m2, D_swap_m1, D_swap_m2,
               validation: "PASS" | "FAIL"}
  sanity_check_report.txt   — human-readable summary for the paper

Usage
-----
  python 04_sanity_self_replace.py \\
      --clips_json      selected_clips/selected_clips.json \\
      --method1_dir     pipeline_output/03a_method1 \\
      --method2_dir     pipeline_output/03b_method2 \\
      --video_dir       /data/videos \\
      --out_dir         pipeline_output/04_sanity \\
      [--world_model    vjepa2]   # or cosmos, mock
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# World model rollout interface (pluggable)
# ---------------------------------------------------------------------------

def _mock_rollout(
    frames_dir: str,
    n_samples: int = 10,
    horizon: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """
    Mock world-model rollout for testing without a real world model.

    Returns (n_samples, horizon, 2) array of (x, y) trajectory waypoints.
    Adds a small random perturbation around a straight-line prior.
    """
    rng = np.random.default_rng(seed)
    base = np.stack(
        [np.linspace(0, 100, horizon), np.linspace(0, 50, horizon)], axis=-1
    )  # (T, 2)
    noise = rng.normal(0, 2.0, (n_samples, horizon, 2))
    return base[None, :, :] + noise   # (n_samples, T, 2)


def run_world_model_rollout(
    frames_dir: str,
    model_name: str = "mock",
    n_samples: int = 10,
    horizon: int = 30,
) -> np.ndarray:
    """
    Run a world model rollout on the frames in frames_dir.

    Returns (n_samples, horizon, D) array where D is the trajectory feature
    dimensionality (at minimum 2: x,y).

    Supported model_name values:
      "mock"   — deterministic random rollout (debug only; never evidence)
      "vjepa2" — V-JEPA 2 rollout (requires vjepa2 package)
      "cosmos" — NVIDIA Cosmos rollout (requires cosmos package)
    """
    if model_name == "mock":
        return _mock_rollout(frames_dir, n_samples, horizon)

    if model_name == "vjepa2":
        try:
            from vjepa2.rollout import rollout_from_frames  # type: ignore
            return rollout_from_frames(frames_dir, n_samples=n_samples, horizon=horizon)
        except ImportError as exc:
            raise RuntimeError("V-JEPA 2 adapter is unavailable; refusing mock evaluation.") from exc

    if model_name == "cosmos":
        try:
            from cosmos.rollout import rollout_from_frames  # type: ignore
            return rollout_from_frames(frames_dir, n_samples=n_samples, horizon=horizon)
        except ImportError as exc:
            raise RuntimeError("Cosmos adapter is unavailable; refusing mock evaluation.") from exc

    raise ValueError(f"Unknown world model: {model_name}")


# ---------------------------------------------------------------------------
# Rollout divergence (simplified scalar for sanity check)
# ---------------------------------------------------------------------------

def trajectory_divergence(
    rollouts_a: np.ndarray,  # (N, T, D)
    rollouts_b: np.ndarray,  # (N, T, D)
) -> float:
    """
    Trajectory divergence: mean Wasserstein-1 proxy between rollout distributions.

    For each time step t, we compute the mean L2 distance between the sample
    means of rollouts_a and rollouts_b.  This is the simplified D_τ proxy
    suitable for the sanity check (not the full DISCRETIO score).

    Returns a scalar divergence value (lower = more similar distributions).
    """
    # Compare means across the horizon
    mean_a = rollouts_a.mean(axis=0)   # (T, D)
    mean_b = rollouts_b.mean(axis=0)   # (T, D)
    diffs  = np.linalg.norm(mean_a - mean_b, axis=-1)   # (T,)
    return float(diffs.mean())


def jsd_divergence(
    rollouts_a: np.ndarray,  # (N, T, D)
    rollouts_b: np.ndarray,  # (N, T, D)
    n_bins: int = 20,
) -> float:
    """
    Simplified JSD between marginal distributions at the final time step.
    Used as a secondary divergence metric.
    """
    final_a = rollouts_a[:, -1, 0]   # x coordinate at t=T
    final_b = rollouts_b[:, -1, 0]

    # Bin both distributions over the same range
    lo = min(final_a.min(), final_b.min()) - 1
    hi = max(final_a.max(), final_b.max()) + 1
    bins = np.linspace(lo, hi, n_bins + 1)

    p, _ = np.histogram(final_a, bins=bins, density=True)
    q, _ = np.histogram(final_b, bins=bins, density=True)

    p = np.clip(p, 1e-10, None); p /= p.sum()
    q = np.clip(q, 1e-10, None); q /= q.sum()
    m = 0.5 * (p + q)

    jsd = 0.5 * (
        np.sum(p * np.log(p / m)) +
        np.sum(q * np.log(q / m))
    )
    return float(jsd)


# ---------------------------------------------------------------------------
# Per-clip sanity check
# ---------------------------------------------------------------------------

def run_sanity_check_clip(
    clip: Dict,
    method1_dir: Path,
    method2_dir: Path,
    video_dir: str,
    world_model: str,
    n_samples: int,
    horizon: int,
    swap_age: str = "child",   # the target age in the actual behavioral prior test
) -> Dict:
    clip_id = clip["clip_id"]
    source_age = clip["age_bracket"]

    # --- O_real: original video frames ---
    orig_frames_dir = _find_original_frames(video_dir, clip)
    if orig_frames_dir is None:
        print(f"  [SKIP] Original frames not found for {clip_id}")
        return {}

    # --- O_recon: same-age (adult→adult) re-inserted frames ---
    recon_m1_dir = method1_dir / clip_id / "method1" / "frames"
    if not recon_m1_dir.exists():
        fallback_recon_m1 = method1_dir.parent / "03a_method1_adult" / clip_id / "method1" / "frames"
        if fallback_recon_m1.exists():
            recon_m1_dir = fallback_recon_m1

    recon_m2_dir = method2_dir / clip_id / "method2" / "frames"
    if not recon_m2_dir.exists():
        fallback_recon_m2 = method2_dir.parent / "03b_method2_adult" / clip_id / "method2" / "frames"
        if fallback_recon_m2.exists():
            recon_m2_dir = fallback_recon_m2

    # --- O_swap: different-age (adult→target) frames for reference D_swap ---
    swap_m1_dir = method1_dir.parent / f"03a_method1_{swap_age}" / clip_id / "method1" / "frames"
    if not swap_m1_dir.exists():
        fallback_swap_m1 = method1_dir / clip_id / "method1" / "frames"
        # Check if method1_dir is already the swap output
        meta_m1 = method1_dir / clip_id / "method1" / "metadata.json"
        if meta_m1.exists():
            try:
                with open(meta_m1) as mf:
                    if json.load(mf).get("target_age") == swap_age:
                        swap_m1_dir = fallback_swap_m1
            except Exception:
                pass

    swap_m2_dir = method2_dir.parent / f"03b_method2_{swap_age}" / clip_id / "method2" / "frames"
    if not swap_m2_dir.exists():
        fallback_swap_m2 = method2_dir / clip_id / "method2" / "frames"
        meta_m2 = method2_dir / clip_id / "method2" / "metadata.json"
        if meta_m2.exists():
            try:
                with open(meta_m2) as mf:
                    if json.load(mf).get("target_age") == swap_age:
                        swap_m2_dir = fallback_swap_m2
            except Exception:
                pass

    print(f"  Running world model ({world_model}) rollouts ...")

    R_real = run_world_model_rollout(str(orig_frames_dir), world_model, n_samples, horizon)

    results: Dict = {
        "clip_id":    clip_id,
        "source_age": source_age,
        "world_model": world_model,
    }

    for method_tag, recon_dir, swap_dir in [
        ("m1", recon_m1_dir, swap_m1_dir),
        ("m2", recon_m2_dir, swap_m2_dir),
    ]:
        if recon_dir.exists():
            R_recon = run_world_model_rollout(str(recon_dir), world_model, n_samples, horizon)
            D_self  = trajectory_divergence(R_real, R_recon)
            JSD_self = jsd_divergence(R_real, R_recon)
        else:
            D_self = JSD_self = float("nan")
            print(f"  [WARN] Recon frames not found ({method_tag}): {recon_dir}")

        if swap_dir.exists():
            R_swap = run_world_model_rollout(str(swap_dir), world_model, n_samples, horizon)
            D_swap = trajectory_divergence(R_real, R_swap)
            JSD_swap = jsd_divergence(R_real, R_swap)
        else:
            D_swap = JSD_swap = float("nan")
            print(f"  [WARN] Swap frames not found ({method_tag}): {swap_dir}")

        # A negligible swap distance makes this ratio undefined as evidence.
        # In particular, identical mock rollouts must never count as a pass.
        if (world_model != "mock" and not (np.isnan(D_self) or np.isnan(D_swap))
                and D_swap > 1e-6):
            ratio    = D_self / (D_swap + 1e-6)
            validate = "PASS" if ratio < 0.3 else "FAIL"
        else:
            ratio    = float("nan")
            validate = "SKIP"

        results[f"D_traj_self_{method_tag}"]   = round(D_self, 4) if not np.isnan(D_self) else None
        results[f"D_traj_swap_{method_tag}"]   = round(D_swap, 4) if not np.isnan(D_swap) else None
        results[f"JSD_self_{method_tag}"]      = round(JSD_self, 4) if not np.isnan(JSD_self) else None
        results[f"JSD_swap_{method_tag}"]      = round(JSD_swap, 4) if not np.isnan(JSD_swap) else None
        results[f"D_self_to_swap_ratio_{method_tag}"] = round(ratio, 4) if not np.isnan(ratio) else None
        results[f"validation_{method_tag}"]    = validate

    overall_pass = all(
        results.get(f"validation_{m}") == "PASS"
        for m in ["m1", "m2"]
    )
    results["overall_validation"] = "PASS" if overall_pass else "FAIL"

    d_self_m1 = results.get("D_traj_self_m1")
    d_self_str = f"{d_self_m1:.3f}" if d_self_m1 is not None and not np.isnan(d_self_m1) else "N/A"
    d_swap_m1 = results.get("D_traj_swap_m1")
    d_swap_str = f"{d_swap_m1:.3f}" if d_swap_m1 is not None and not np.isnan(d_swap_m1) else "N/A"
    print(f"  {clip_id}: D_self_m1={d_self_str}  D_swap_m1={d_swap_str}  → {results.get('validation_m1', '?')}")

    return results


def _find_original_frames(video_dir: str, clip: Dict) -> Optional[Path]:
    """
    Locate the extracted original frames for a clip.
    If not pre-extracted, extract them to a tmp location (or return None).
    """
    clip_id  = clip["clip_id"]
    # Check if Step 1 already saved raw frames
    for cand in [
        Path("pipeline_output/01_masks") / clip_id / "raw_frames",
        Path("pipeline_output/03b_method2") / clip_id / "raw_frames",
        Path("/tmp") / f"discretio_orig_{clip_id}",
    ]:
        if cand.exists() and any(cand.glob("frame_*.png")):
            return cand

    video_dir_p = Path(video_dir)
    video_id = clip["video_id"]
    dataset  = clip.get("dataset", "")
    start_f  = int(clip["start_frame"])
    end_f    = int(clip["end_frame"])

    names_to_try = []
    if clip.get("local_video"):
        names_to_try.append(clip["local_video"])
    for ext in (".mp4", ".avi", ".mov", ".MP4"):
        names_to_try.append(f"{clip_id}{ext}")
        names_to_try.append(f"{video_id}{ext}")

    roots = [video_dir_p]
    if dataset:
        roots.append(video_dir_p / dataset)

    for root in roots:
        for name in names_to_try:
            vp = root / name
            if vp.is_file():
                tmp_dir = Path("/tmp") / f"discretio_orig_{clip_id}"
                if not tmp_dir.exists() or not any(tmp_dir.glob("frame_*.png")):
                    from utils.video_io import extract_frames
                    extract_frames(str(vp), str(tmp_dir), start_frame=start_f, end_frame=end_f)
                return tmp_dir
        for p in root.glob(f"*{video_id}*.mp4"):
            if p.is_file():
                tmp_dir = Path("/tmp") / f"discretio_orig_{clip_id}"
                if not tmp_dir.exists() or not any(tmp_dir.glob("frame_*.png")):
                    from utils.video_io import extract_frames
                    extract_frames(str(p), str(tmp_dir), start_frame=start_f, end_frame=end_f)
                return tmp_dir
    return None


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def write_report(results: List[Dict], out_path: Path) -> None:
    lines = [
        "DISCRETIO — Self-Replacement Sanity Check Report",
        "=" * 60,
        f"World model: {results[0].get('world_model', '?') if results else '?'}",
        f"Clips evaluated: {len(results)}",
        "",
        "Criterion: D_self / D_swap < 0.3  →  PASS",
        "",
    ]
    n_pass = sum(1 for r in results if r.get("overall_validation") == "PASS")
    n_fail = sum(1 for r in results if r.get("overall_validation") == "FAIL")

    for r in results:
        lines.append(
            f"  {r['clip_id']:50s} "
            f"M1={r.get('validation_m1','?'):4s}  "
            f"M2={r.get('validation_m2','?'):4s}  "
            f"→ {r.get('overall_validation','?')}"
        )

    lines += [
        "",
        f"Overall: {n_pass}/{len(results)} PASS, {n_fail}/{len(results)} FAIL",
        "",
        "Interpretation for paper §Quality Control:",
        "  PASS: pipeline noise is << actor-identity signal  (publishable)",
        "  FAIL: revisit inpainting quality or world model rollout variance",
    ]
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Self-Replacement Sanity Check for the DISCRETIO pipeline."
    )
    parser.add_argument("--clips_json",   required=True)
    parser.add_argument("--method1_dir",  default="pipeline_output/03a_method1")
    parser.add_argument("--method2_dir",  default="pipeline_output/03b_method2")
    parser.add_argument("--video_dir",    required=True)
    parser.add_argument("--out_dir",      default="pipeline_output/04_sanity")
    parser.add_argument("--world_model",  default="mock",
                        choices=["mock", "vjepa2", "cosmos"],
                        help="World model to use for rollout generation.")
    parser.add_argument("--n_samples",    type=int, default=10)
    parser.add_argument("--horizon",      type=int, default=30)
    parser.add_argument("--swap_age",     default="child",
                        choices=["child", "adult", "senior"])
    args = parser.parse_args()

    with open(args.clips_json) as f:
        clips = json.load(f)

    # Only process sanity_check clips (adult-only, role="sanity_check")
    sanity_clips = [c for c in clips if c.get("role") == "sanity_check"]
    if not sanity_clips:
        print("No sanity_check clips found in clips_json. "
              "Make sure 00_select_clips.py assigned roles correctly.")
        return

    print(f"Running sanity check on {len(sanity_clips)} clips with '{args.world_model}'.\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for clip in sanity_clips:
        print(f"\n[Sanity] {clip['clip_id']}")
        r = run_sanity_check_clip(
            clip,
            Path(args.method1_dir),
            Path(args.method2_dir),
            args.video_dir,
            args.world_model,
            args.n_samples,
            args.horizon,
            args.swap_age,
        )
        if r:
            all_results.append(r)

    # Save JSON results
    json_out = out_dir / "sanity_check_results.json"
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=2)

    # Write human-readable report
    report_out = out_dir / "sanity_check_report.txt"
    write_report(all_results, report_out)

    print(f"\nResults saved to: {json_out}")
    print(f"Report saved to:  {report_out}")
    print()
    n_pass = sum(1 for r in all_results if r.get("overall_validation") == "PASS")
    print(f"{'='*50}")
    print(f"SANITY CHECK: {n_pass}/{len(all_results)} clips PASSED.")
    if n_pass == len(all_results):
        print("✓ Sanity criterion passed on evaluated clips.")
    else:
        print("✗ Some clips FAILED — review inpainting quality before proceeding.")


if __name__ == "__main__":
    main()
