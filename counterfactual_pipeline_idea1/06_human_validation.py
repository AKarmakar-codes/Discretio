#!/usr/bin/env python3
"""
06_human_validation.py
-----------------------
Step 6 — Human Validation Score (Table 6 in the DISCRETIO paper).

Implements the full human validation protocol described in the plan:
  - 3–5 lab members blind-rate 10 clips on a 1–5 Likert scale.
  - Three dimensions: Realism, Factor Isolation, Absence of Artifacts.
  - Fleiss' κ computed for inter-annotator agreement.

This script has two modes:
  1. GENERATE: Produce annotator input sheets (CSV) with clip IDs, blinded
     labels, and instructions. Annotators fill in their scores offline.
  2. AGGREGATE: Read filled-in annotator CSVs, compute per-clip means,
     per-dimension Fleiss' κ, and write Table 6 data for the paper.

Table 6 output columns (paper §3.5 / §Quality Control):
  clip_id | method | realism_mean ± std | factor_iso_mean ± std |
  artifact_free_mean ± std | fleiss_kappa_realism | fleiss_kappa_factor_iso |
  fleiss_kappa_artifact | overall_pass

Usage — Generate sheets:
  python 06_human_validation.py generate \\
      --clips_json      selected_clips/selected_clips.json \\
      --method1_dir     pipeline_output/03a_method1 \\
      --method2_dir     pipeline_output/03b_method2 \\
      --out_dir         pipeline_output/06_human_val \\
      --n_annotators    4

Usage — Aggregate results:
  python 06_human_validation.py aggregate \\
      --filled_csvs     ann1.csv ann2.csv ann3.csv ann4.csv \\
      --out_dir         pipeline_output/06_human_val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Likert scale definitions
# ---------------------------------------------------------------------------
LIKERT_SCALE = {1: "Very Poor", 2: "Poor", 3: "Acceptable", 4: "Good", 5: "Excellent"}
DIMENSIONS   = ["realism", "factor_isolation", "absence_of_artifacts"]
MIN_KAPPA_PASS = 0.60   # Fleiss' κ ≥ 0.60 = "substantial agreement" (Landis & Koch)


# ---------------------------------------------------------------------------
# Fleiss' κ
# ---------------------------------------------------------------------------

def fleiss_kappa(ratings_matrix: np.ndarray, n_categories: int = 5) -> float:
    """
    Compute Fleiss' κ from a ratings matrix.

    Parameters
    ----------
    ratings_matrix : (N_items, N_annotators) int array of Likert ratings (1–5)
    n_categories   : number of rating categories (5 for 1–5 Likert)

    Returns
    -------
    kappa : float
    """
    N, k = ratings_matrix.shape  # items × annotators
    categories = list(range(1, n_categories + 1))

    # Build category count matrix (N × n_categories)
    n_j = np.zeros((N, n_categories), dtype=np.float64)
    for j, cat in enumerate(categories):
        n_j[:, j] = (ratings_matrix == cat).sum(axis=1)

    # Overall proportion for each category
    p_j = n_j.sum(axis=0) / (N * k)

    # Per-item agreement
    P_i = ((n_j ** 2).sum(axis=1) - k) / (k * (k - 1))

    # Expected agreement
    P_bar = P_i.mean()
    P_e   = (p_j ** 2).sum()

    if abs(1.0 - P_e) < 1e-10:
        return 1.0
    return float((P_bar - P_e) / (1.0 - P_e))


# ---------------------------------------------------------------------------
# Mode 1: Generate annotator sheets
# ---------------------------------------------------------------------------

def generate_sheets(
    clips: List[Dict],
    method1_dir: Path,
    method2_dir: Path,
    out_dir: Path,
    n_annotators: int,
    target_age: str,
) -> None:
    """
    For each clip × method, create a blind entry in the annotator CSV.
    Clips are shuffled and given opaque IDs to prevent bias.
    """
    import random
    rng = random.Random(42)

    # Build entries: (blind_id, real_clip_id, method, video_path)
    entries = []
    for clip in clips:
        cid = clip["clip_id"]
        for method_tag, method_dir in [("m1", method1_dir), ("m2", method2_dir)]:
            frames_subdir = "method1" if method_tag == "m1" else "method2"
            vdir = method_dir / cid / frames_subdir / "frames"
            entries.append({
                "blind_id":    None,
                "clip_id":     cid,
                "method_tag":  method_tag,
                "frames_path": str(vdir),
                "source_age":  clip["age_bracket"],
                "target_age":  target_age,
            })

    rng.shuffle(entries)
    for i, e in enumerate(entries):
        e["blind_id"] = f"CLIP_{i + 1:03d}"

    # Save key (not shown to annotators)
    key_path = out_dir / "blind_key.json"
    with open(key_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Blind key (keep private): {key_path}")

    # Annotator instructions
    instructions = """
DISCRETIO Human Validation — Annotator Instructions
====================================================

You will evaluate {n} video clips of pedestrian scenes.
Each clip is a modified version of an original traffic video.

For each clip, rate on a scale of 1–5:

  1. REALISM
     How natural and realistic does the edited pedestrian look?
     1 = Obviously fake / heavily distorted
     5 = Indistinguishable from a real person

  2. FACTOR ISOLATION
     Does the pedestrian's POSITION, MOTION, and SCENE CONTEXT appear
     identical to what you would expect from the original clip?
     (Only the person's apparent age should change, nothing else.)
     1 = Many other things changed (background, motion, scene)
     5 = Only the person's age appears different

  3. ABSENCE OF ARTIFACTS
     Are there visual artifacts (blurring, ghosting, flickering,
     boundary glitches, unrealistic lighting)?
     1 = Severe artifacts throughout
     5 = No visible artifacts

Instructions:
  - Fill in ONLY columns realism, factor_isolation, absence_of_artifacts.
  - Use integers 1–5.
  - Do NOT modify the blind_id column.
  - Save as: annotator_<your_name>.csv
""".format(n=len(entries))

    inst_path = out_dir / "annotator_instructions.txt"
    inst_path.write_text(instructions)
    print(f"Instructions: {inst_path}")

    # Generate per-annotator CSV sheets
    for ann_idx in range(1, n_annotators + 1):
        rows = []
        for e in entries:
            rows.append({
                "blind_id":             e["blind_id"],
                "frames_path":          e["frames_path"],   # annotator views this
                "realism":              "",
                "factor_isolation":     "",
                "absence_of_artifacts": "",
            })
        df = pd.DataFrame(rows)
        sheet_path = out_dir / f"annotator_{ann_idx:02d}_sheet.csv"
        df.to_csv(sheet_path, index=False)
        print(f"  Annotator {ann_idx} sheet: {sheet_path}")

    print(f"\nGenerated {n_annotators} annotator sheets for {len(entries)} blind clips.")
    print("Distribute sheets + instructions to annotators.")


# ---------------------------------------------------------------------------
# Mode 2: Aggregate filled-in sheets → Table 6
# ---------------------------------------------------------------------------

def aggregate_results(
    filled_csvs: List[str],
    blind_key_path: str,
    out_dir: Path,
) -> None:
    """
    Read filled-in annotator CSVs, join to blind key, compute stats and κ.
    """
    # Load blind key
    with open(blind_key_path) as f:
        key_entries = json.load(f)
    key_df = pd.DataFrame(key_entries)[["blind_id", "clip_id", "method_tag"]]

    # Load and stack annotator responses
    ann_dfs = []
    for path in filled_csvs:
        df = pd.read_csv(path)
        required = {"blind_id"} | set(DIMENSIONS)
        missing  = required - set(df.columns)
        if missing:
            print(f"[WARN] {path} is missing columns: {missing}. Skipping.")
            continue
        df = df[["blind_id"] + DIMENSIONS].copy()
        for dim in DIMENSIONS:
            df[dim] = pd.to_numeric(df[dim], errors="coerce")
        ann_dfs.append(df)

    if not ann_dfs:
        print("ERROR: No valid annotator CSVs loaded.")
        sys.exit(1)

    n_ann = len(ann_dfs)
    print(f"Loaded {n_ann} annotator response sheets.")

    # Pivot: blind_id × annotator → dimension score
    all_rows = []
    for clip_entry in key_entries:
        bid = clip_entry["blind_id"]
        row: Dict = {
            "blind_id":  bid,
            "clip_id":   clip_entry["clip_id"],
            "method_tag": clip_entry["method_tag"],
        }
        for dim in DIMENSIONS:
            scores = []
            for df in ann_dfs:
                match = df[df["blind_id"] == bid][dim]
                if not match.empty and not np.isnan(match.iloc[0]):
                    scores.append(match.iloc[0])
            row[f"{dim}_scores"] = scores
            row[f"{dim}_mean"]   = float(np.mean(scores)) if scores else None
            row[f"{dim}_std"]    = float(np.std(scores))  if scores else None
        all_rows.append(row)

    # Compute Fleiss' κ per dimension per method
    kappa_results: Dict = {}
    for method_tag in ["m1", "m2"]:
        method_rows = [r for r in all_rows if r["method_tag"] == method_tag]
        for dim in DIMENSIONS:
            # Build ratings matrix (N_items × N_annotators)
            mat = []
            for r in method_rows:
                scores = r[f"{dim}_scores"]
                if len(scores) == n_ann:
                    mat.append(scores)
            if mat:
                mat_arr = np.array(mat, dtype=int)
                kappa   = fleiss_kappa(mat_arr, n_categories=5)
            else:
                kappa = None
            kappa_results[f"{method_tag}_{dim}_kappa"] = kappa

    # Build Table 6
    table_rows = []
    for r in all_rows:
        mt = r["method_tag"]
        table_row = {
            "clip_id":    r["clip_id"],
            "method":     mt,
        }
        for dim in DIMENSIONS:
            table_row[f"{dim}_mean"] = r.get(f"{dim}_mean")
            table_row[f"{dim}_std"]  = r.get(f"{dim}_std")
            table_row[f"{dim}_kappa"] = kappa_results.get(f"{mt}_{dim}_kappa")

        # Clip passes if all dimension means ≥ 3.5 and κ ≥ MIN_KAPPA_PASS
        means_ok   = all(
            (table_row.get(f"{d}_mean") or 0) >= 3.5 for d in DIMENSIONS
        )
        kappas_ok  = all(
            (table_row.get(f"{d}_kappa") is not None and
             table_row.get(f"{d}_kappa", 0) >= MIN_KAPPA_PASS)
            for d in DIMENSIONS
        )
        table_row["overall_pass"] = means_ok and kappas_ok
        table_rows.append(table_row)

    table_df = pd.DataFrame(table_rows)

    # Save outputs
    csv_out  = out_dir / "table6_human_validation.csv"
    json_out = out_dir / "table6_human_validation.json"
    table_df.to_csv(csv_out,  index=False)
    table_df.to_json(json_out, orient="records", indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("TABLE 6 — Human Validation Summary")
    print("=" * 70)
    for method_tag, label in [("m1", "Method 1 (Erase + Pose Retarget)"),
                               ("m2", "Method 2 (Unified Inpaint)")]:
        m_rows = table_df[table_df["method"] == method_tag]
        pass_n = m_rows["overall_pass"].sum()
        print(f"\n  {label}:")
        for dim in DIMENSIONS:
            mean_col  = f"{dim}_mean"
            kappa_col = f"{dim}_kappa"
            mean_val  = m_rows[mean_col].mean() if (not m_rows.empty and mean_col in m_rows) else None
            kappa_val = m_rows[kappa_col].iloc[0] if (not m_rows.empty and kappa_col in m_rows.columns) else None
            mean_s  = f"{mean_val:.2f}" if (mean_val is not None and not np.isnan(mean_val)) else "N/A"
            kappa_s = f"{kappa_val:.3f}" if (kappa_val is not None and not np.isnan(kappa_val)) else "N/A"
            print(f"    {dim:<30}: mean={mean_s}  κ={kappa_s}")
        print(f"    Clips passing: {pass_n}/{len(m_rows)}")

    # Overall winner
    m1_pass = table_df[table_df["method"] == "m1"]["overall_pass"].sum()
    m2_pass = table_df[table_df["method"] == "m2"]["overall_pass"].sum()
    winner  = "Method 1" if m1_pass > m2_pass else ("Method 2" if m2_pass > m1_pass else "No winner (tie / insufficient evidence)")
    print(f"\n{'='*70}")
    print(f"HUMAN-RATING COMPARISON (pilot; not automatic benchmark validation): {winner}")
    print(f"  → Use this method for all DISCRETIO behavioral-prior probes.")
    print(f"{'='*70}")

    print(f"\nTable 6 saved to: {csv_out}")

    # Also save kappa summary
    kappa_df = pd.DataFrame([{
        "method_dim": k, "fleiss_kappa": v
    } for k, v in kappa_results.items()])
    kappa_df.to_csv(out_dir / "fleiss_kappa.csv", index=False)
    print(f"Fleiss' κ saved to: {out_dir / 'fleiss_kappa.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Human validation protocol for DISCRETIO counterfactual pipeline (Table 6)."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- generate ---
    gen_p = sub.add_parser("generate", help="Generate annotator input sheets.")
    gen_p.add_argument("--clips_json",   required=True)
    gen_p.add_argument("--method1_dir",  default="pipeline_output/03a_method1")
    gen_p.add_argument("--method2_dir",  default="pipeline_output/03b_method2")
    gen_p.add_argument("--out_dir",      default="pipeline_output/06_human_val")
    gen_p.add_argument("--n_annotators", type=int, default=4)
    gen_p.add_argument("--target_age", choices=["child", "adult", "senior"], default="adult")

    # --- aggregate ---
    agg_p = sub.add_parser("aggregate", help="Aggregate filled annotator CSVs into Table 6.")
    agg_p.add_argument("--filled_csvs",  nargs="+", required=True,
                        help="Paths to filled-in annotator CSV files.")
    agg_p.add_argument("--blind_key",    default="pipeline_output/06_human_val/blind_key.json")
    agg_p.add_argument("--out_dir",      default="pipeline_output/06_human_val")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "generate":
        with open(args.clips_json) as f:
            clips = json.load(f)
        generate_sheets(
            clips,
            Path(args.method1_dir),
            Path(args.method2_dir),
            out_dir,
            args.n_annotators,
            args.target_age,
        )
    elif args.mode == "aggregate":
        aggregate_results(args.filled_csvs, args.blind_key, out_dir)


if __name__ == "__main__":
    main()
