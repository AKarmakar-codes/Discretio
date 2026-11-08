#!/usr/bin/env python3
"""
00_select_clips.py
------------------
Step 0 of the DISCRETIO counterfactual pipeline.

Mines 10 benchmark clips from IDD-PeD only for the empirical method
selection experiment described in the paper (§Quality Control and Human
Validation, Table 6).

Selection criteria
------------------
  - Clip must contain a pedestrian whose age_bracket is clearly annotated
    as "child", "adult", or "senior" (not "default").
  - t0 anchor frame is at least T_CTX_FRAMES (60) from the start of the
    clip so the full context window exists.
  - Pedestrian must be unoccluded (or only partially occluded) at t0.
  - Solo pedestrian preferred (is_solo == True) to minimise confounds.
  - Diversity: at most 4 clips per age bracket to ensure age spread across
    the 10 clips (child / adult / senior).

Outputs
-------
  selected_clips.json   —  list of dicts with keys:
      clip_id, dataset, video_id, track_id, t0_frame, start_frame,
      end_frame, age_bracket, bbox_t0, outcome, role
  selected_clips.csv    —  same data as CSV for quick inspection

The "role" field is "sanity_check" for adult clips used in the
Self-Replacement Sanity Check (Step 4), and "behavioral_prior" for all
others.  At most 2 adult clips are marked as sanity_check; remaining
adults become behavioral_prior.

Usage
-----
  python 00_select_clips.py \\
      --idd_manifest path/to/idd_ped_master_manifest.csv \\
      --out_dir       ./selected_clips
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_CLIPS_TOTAL    = 10   # all clips come from IDD-PeD
MAX_PER_AGE      = 4    # allow up to 4 per bracket so 10 can be filled
MAX_SANITY_CHECK = 2    # adult clips reserved for self-replacement sanity check

# Occlusion states allowed at t0
ALLOWED_OCCLUSION = {"none", "partial"}

# Columns required in the IDD-PeD manifest CSV
REQUIRED_COLUMNS = {
    "video_id", "track_id", "t0_frame", "start_frame", "end_frame",
    "age_bracket", "xtl_t0", "ytl_t0", "xbr_t0", "ybr_t0",
    "outcome", "is_solo", "occlusion_t0",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> pd.DataFrame:
    """Load the IDD-PeD manifest CSV and validate required columns."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"IDD-PeD manifest is missing columns: {missing}\n"
            f"  Found: {list(df.columns)}"
        )
    df["_dataset"] = "IDD-PeD"
    return df


def filter_manifest(df: pd.DataFrame, min_t0_buffer: int = 60) -> pd.DataFrame:
    """
    Apply clip selection criteria:
      - age_bracket in {child, adult, senior}
      - occlusion_t0 in ALLOWED_OCCLUSION
      - t0_frame >= start_frame + min_t0_buffer (full context window exists)
      - outcome != 'censored_truncated'
    """
    df = df[df["age_bracket"].isin({"child", "adult", "senior"})].copy()
    df = df[df["occlusion_t0"].isin(ALLOWED_OCCLUSION)].copy()
    df = df[df["t0_frame"] >= df["start_frame"] + min_t0_buffer].copy()
    df = df[df["outcome"] != "censored_truncated"].copy()
    return df.reset_index(drop=True)


def select_diverse(df: pd.DataFrame, n: int, max_per_age: int) -> pd.DataFrame:
    """
    Select up to `n` clips with at most `max_per_age` per age bracket.
    Preference: solo pedestrians first, then alphabetically by bracket for
    reproducibility.
    """
    # A stable tie-break is essential: pandas' default sort can pick a
    # different set after row order changes. Deduplicate scene/track anchors.
    df = df.sort_values(
        ["is_solo", "age_bracket", "video_id", "track_id", "t0_frame"],
        ascending=[False, True, True, True, True], kind="stable",
    ).drop_duplicates(["video_id", "track_id", "t0_frame"])

    selected_rows = []
    age_counts: dict = {}

    for _, row in df.iterrows():
        age = row["age_bracket"]
        if age_counts.get(age, 0) >= max_per_age:
            continue
        selected_rows.append(row)
        age_counts[age] = age_counts.get(age, 0) + 1
        if len(selected_rows) >= n:
            break

    if not selected_rows:
        return pd.DataFrame()

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def assign_roles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark up to MAX_SANITY_CHECK adult clips as 'sanity_check';
    all remaining rows get role 'behavioral_prior'.
    """
    roles = []
    sanity_count = 0
    for _, row in df.iterrows():
        if row["age_bracket"] == "adult" and sanity_count < MAX_SANITY_CHECK:
            roles.append("sanity_check")
            sanity_count += 1
        else:
            roles.append("behavioral_prior")
    df = df.copy()
    df["role"] = roles
    return df


def build_clip_id(row: pd.Series) -> str:
    """Construct a stable, unique clip identifier from row fields."""
    return f"IDD-PeD_{row['video_id']}_{row['track_id']}_t{int(row['t0_frame']):06d}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Select 10 IDD-PeD benchmark clips for the DISCRETIO "
                    "empirical method selection experiment."
    )
    parser.add_argument("--idd_manifest", required=True,
                        help="Path to idd_ped_master_manifest.csv")
    parser.add_argument("--out_dir", default="selected_clips",
                        help="Directory to write selected_clips.json and .csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load & filter ----
    print(f"Loading IDD-PeD manifest: {args.idd_manifest}")
    df = load_manifest(args.idd_manifest)
    filtered = filter_manifest(df)
    print(f"  Candidates after filtering: {len(filtered)}")

    if len(filtered) == 0:
        print("ERROR: No candidates passed the filter. Check your manifest.")
        sys.exit(1)

    # ---- Select 10 diverse clips ----
    selected = select_diverse(filtered, N_CLIPS_TOTAL, MAX_PER_AGE)
    print(f"  Selected: {len(selected)}")

    if len(selected) < N_CLIPS_TOTAL:
        print(f"  [WARN] Only {len(selected)} clips available after diversity filter "
              f"(target: {N_CLIPS_TOTAL}).")
        # Preserve the age cap and reported criteria; silently relaxing either
        # would change the sampling protocol.
        print("  [ERROR] Cannot fill target under the age cap; lower the target or revise the cap.")
        sys.exit(2)

    selected = assign_roles(selected)

    # ---- Build clip_id from row fields (avoid pandas attribute access) ----
    selected["clip_id"] = selected.apply(build_clip_id, axis=1)

    # ---- Subset output columns ----
    out_cols = [
        "clip_id", "_dataset", "video_id", "track_id",
        "t0_frame", "start_frame", "end_frame",
        "age_bracket", "xtl_t0", "ytl_t0", "xbr_t0", "ybr_t0",
        "outcome", "role",
    ]
    out_df = selected[out_cols].rename(columns={"_dataset": "dataset"})

    # ---- Save ----
    csv_path  = out_dir / "selected_clips.csv"
    json_path = out_dir / "selected_clips.json"
    out_df.to_csv(csv_path, index=False)
    out_df.to_json(json_path, orient="records", indent=2)

    print("\n" + "=" * 60)
    print("SELECTED CLIPS SUMMARY  (IDD-PeD only)")
    print("=" * 60)
    print(f"  Total clips selected : {len(out_df)}")
    print(f"\n  By age bracket:")
    print(out_df["age_bracket"].value_counts().to_string())
    print(f"\n  By role:")
    print(out_df["role"].value_counts().to_string())
    print(f"\nOutputs written to:")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
