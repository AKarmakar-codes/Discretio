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
import os
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


def filter_manifest(df: pd.DataFrame, min_t0_buffer: int = 60, strict: bool = True) -> pd.DataFrame:
    """
    Apply clip selection criteria:
      - age_bracket in {child, adult, senior}
      - occlusion_t0 in ALLOWED_OCCLUSION
      - t0_frame >= start_frame + min_t0_buffer (full context window exists)
      - outcome != 'censored_truncated' (relaxed if strict=False)
    """
    out = df[df["age_bracket"].isin({"child", "adult", "senior"})].copy()
    if strict:
        out = out[out["occlusion_t0"].isin(ALLOWED_OCCLUSION)].copy()
        out = out[out["t0_frame"] >= out["start_frame"] + min_t0_buffer].copy()
        out = out[out["outcome"] != "censored_truncated"].copy()
    return out.reset_index(drop=True)


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

def resolve_manifest_path(user_path: str | None) -> str:
    """Find the manifest file across common paths if user path is missing."""
    candidates = []
    if user_path:
        candidates.append(user_path)
    candidates.extend([
        "idd_ped_master_manifest.csv",
        "../idd_ped_master_manifest.csv",
        "Discretio/idd_ped_master_manifest.csv",
        "/kaggle/working/Discretio/idd_ped_master_manifest.csv",
        "/kaggle/working/idd_ped_master_manifest.csv",
        "/kaggle/input/idd-ped-samples/idd_ped_master_manifest.csv",
    ])
    for c in candidates:
        if c and os.path.isfile(c):
            return str(Path(c).resolve())
    raise FileNotFoundError(
        f"Could not find IDD-PeD manifest file at '{user_path}'.\n"
        f"Searched candidates:\n  " + "\n  ".join(candidates)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Select benchmark clips for the DISCRETIO empirical method selection experiment."
    )
    parser.add_argument("--idd_manifest", default=None,
                        help="Path to idd_ped_master_manifest.csv (auto-detected if omitted).")
    parser.add_argument("--n_clips", type=int, default=10,
                        help="Target number of clips to select (default: 10).")
    parser.add_argument("--allow_fewer", action="store_true",
                        help="Allow selecting fewer clips if manifest has limited samples.")
    parser.add_argument("--out_dir", default="selected_clips",
                        help="Directory to write selected_clips.json and .csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = resolve_manifest_path(args.idd_manifest)
    print(f"Loading IDD-PeD manifest: {manifest_path}")
    df = load_manifest(manifest_path)
    print(f"  Raw events in manifest: {len(df)}")

    filtered = filter_manifest(df, strict=True)
    print(f"  Candidates after strict filtering: {len(filtered)}")

    if len(filtered) == 0:
        if args.allow_fewer:
            print("  [WARN] 0 candidates passed strict filter. Relaxing to un-truncated events.")
            filtered = filter_manifest(df, strict=False)
            print(f"  Candidates after relaxed filtering: {len(filtered)}")
        if len(filtered) == 0:
            print("ERROR: No candidates passed the filter. Check your manifest contents.")
            sys.exit(1)

    target_n = min(args.n_clips, len(filtered)) if args.allow_fewer else args.n_clips
    max_per_age = max(1, target_n // 2)

    selected = select_diverse(filtered, target_n, max_per_age)
    print(f"  Selected: {len(selected)} (target: {target_n})")

    if len(selected) < target_n:
        if args.allow_fewer:
            print(f"  [INFO] Selected {len(selected)} clips under --allow_fewer.")
        else:
            print(f"  [ERROR] Only {len(selected)} clips available (target: {target_n}). "
                  f"Use --allow_fewer or provide a larger manifest.")
            sys.exit(2)

    selected = assign_roles(selected)
    selected["clip_id"] = selected.apply(build_clip_id, axis=1)

    out_cols = [
        "clip_id", "_dataset", "video_id", "track_id",
        "t0_frame", "start_frame", "end_frame",
        "age_bracket", "xtl_t0", "ytl_t0", "xbr_t0", "ybr_t0",
        "outcome", "role",
    ]
    out_df = selected[out_cols].rename(columns={"_dataset": "dataset"})

    csv_path  = out_dir / "selected_clips.csv"
    json_path = out_dir / "selected_clips.json"
    out_df.to_csv(csv_path, index=False)
    out_df.to_json(json_path, orient="records", indent=2)

    print("\n" + "=" * 60)
    print("SELECTED CLIPS SUMMARY (IDD-PeD only)")
    print("=" * 60)
    print(f"  Total clips selected : {len(out_df)}")
    print(f"\n  By age bracket:")
    print(out_df["age_bracket"].value_counts().to_string())
    print(f"\n  By role:")
    print(out_df["role"].value_counts().to_string())
    print(f"\nOutputs successfully written to:")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
