#!/usr/bin/env bash
# run_pipeline.sh
# End-to-end driver for the DISCRETIO child↔adult counterfactual pipeline.
# Edit the variables below before running.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# USER CONFIG — Edit these paths
# ============================================================
CLIPS_JSON="selected_clips/selected_clips.json"
IDD_MANIFEST="/kaggle/input/idd-ped/idd_ped_master_manifest.csv"
VIDEO_DIR="/data/videos"
WAN_CHECKPOINT="Wan2.1-VACE-14B"
PROPAINTER_DIR="/path/to/ProPainter"
WORLD_MODEL="vjepa2"   # requires a real adapter; mock rollouts are not evidence
TARGET_AGE="adult"   # child→adult intervention used in the paper's example
N_ANNOTATORS=4

OUT_BASE="pipeline_output"
# ============================================================

if [ -f "$CLIPS_JSON" ]; then
    echo "===== STEP 0: Using existing clip manifest ($CLIPS_JSON) ====="
else
    echo "===== STEP 0: Select 10 IDD-PeD benchmark clips ====="
    python 00_select_clips.py \
        --idd_manifest  "$IDD_MANIFEST" \
        --out_dir       selected_clips
fi

echo ""
echo "===== STEP 1: Segment pedestrians (SAM 2) ====="
python 01_segment_pedestrian.py \
    --clips_json  "$CLIPS_JSON" \
    --video_dir   "$VIDEO_DIR" \
    --out_dir     "$OUT_BASE/01_masks"

echo ""
echo "===== STEP 2: Erase pedestrians (ProPainter) ====="
python 02_erase_pedestrian.py \
    --clips_json     "$CLIPS_JSON" \
    --masks_dir      "$OUT_BASE/01_masks" \
    --video_dir      "$VIDEO_DIR" \
    --out_dir        "$OUT_BASE/02_erased" \
    --propainter_dir "$PROPAINTER_DIR"

echo ""
echo "===== STEP 3a: Method 1 — Erase + Pose Retarget ====="
python 03a_pose_retarget.py \
    --clips_json     "$CLIPS_JSON" \
    --masks_dir      "$OUT_BASE/01_masks" \
    --erased_dir     "$OUT_BASE/02_erased" \
    --video_dir      "$VIDEO_DIR" \
    --out_dir        "$OUT_BASE/03a_method1" \
    --target_age     "$TARGET_AGE" \
    --wan_checkpoint "$WAN_CHECKPOINT"

echo ""
echo "===== STEP 3b: Method 2 — Unified Mask-Box Inpainting ====="
python 03b_unified_inpaint.py \
    --clips_json     "$CLIPS_JSON" \
    --masks_dir      "$OUT_BASE/01_masks" \
    --video_dir      "$VIDEO_DIR" \
    --out_dir        "$OUT_BASE/03b_method2" \
    --target_age     "$TARGET_AGE" \
    --wan_checkpoint "$WAN_CHECKPOINT"

echo ""
echo "===== Generate child counterfactuals for the adult self-replacement control ====="
python 03a_pose_retarget.py \
    --clips_json     "$CLIPS_JSON" \
    --masks_dir      "$OUT_BASE/01_masks" \
    --erased_dir     "$OUT_BASE/02_erased" \
    --video_dir      "$VIDEO_DIR" \
    --out_dir        "$OUT_BASE/03a_method1_child" \
    --target_age     child \
    --wan_checkpoint "$WAN_CHECKPOINT"
python 03b_unified_inpaint.py \
    --clips_json     "$CLIPS_JSON" \
    --masks_dir      "$OUT_BASE/01_masks" \
    --video_dir      "$VIDEO_DIR" \
    --out_dir        "$OUT_BASE/03b_method2_child" \
    --target_age     child \
    --wan_checkpoint "$WAN_CHECKPOINT"

echo ""
echo "===== STEP 4: Sanity Self-Replacement Check ====="
python 04_sanity_self_replace.py \
    --clips_json  "$CLIPS_JSON" \
    --method1_dir "$OUT_BASE/03a_method1" \
    --method2_dir "$OUT_BASE/03b_method2" \
    --video_dir   "$VIDEO_DIR" \
    --out_dir     "$OUT_BASE/04_sanity" \
    --world_model "$WORLD_MODEL"

echo ""
echo "===== Checking sanity result before proceeding ====="
python - <<'EOF'
import json, sys
with open("pipeline_output/04_sanity/sanity_check_results.json") as f:
    results = json.load(f)
n_fail = sum(1 for r in results if r.get("overall_validation") == "FAIL")
if n_fail > 0:
    print(f"ERROR: {n_fail} sanity check(s) FAILED. Fix before proceeding.")
    sys.exit(1)
print(f"All {len(results)} sanity checks PASSED. Proceeding...")
EOF

echo ""
echo "===== STEP 5: Automated Metrics ====="
python 05_metrics_auto.py \
    --clips_json  "$CLIPS_JSON" \
    --method1_dir "$OUT_BASE/03a_method1" \
    --method2_dir "$OUT_BASE/03b_method2" \
    --masks_dir   "$OUT_BASE/01_masks" \
    --video_dir   "$VIDEO_DIR" \
    --out_dir     "$OUT_BASE/05_metrics" \
    --target_age  "$TARGET_AGE"

echo ""
echo "===== STEP 6a: Generate Human Validation Sheets ====="
python 06_human_validation.py generate \
    --clips_json   "$CLIPS_JSON" \
    --method1_dir  "$OUT_BASE/03a_method1" \
    --method2_dir  "$OUT_BASE/03b_method2" \
    --out_dir      "$OUT_BASE/06_human_val" \
    --n_annotators "$N_ANNOTATORS" \
    --target_age   "$TARGET_AGE"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE (through automated metrics)."
echo ""
echo "Next steps:"
echo "  1. Distribute annotator sheets from: $OUT_BASE/06_human_val/"
echo "  2. After annotation, run:"
echo "     python 06_human_validation.py aggregate \\"
echo "       --filled_csvs annotator_01_sheet.csv annotator_02_sheet.csv ... \\"
echo "       --out_dir $OUT_BASE/06_human_val"
echo "  3. Use winning method for all behavioral-prior probes."
echo "============================================================"
