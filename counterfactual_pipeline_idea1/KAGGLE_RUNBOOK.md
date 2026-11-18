# Kaggle workflow: child/adult stimulus pilot

## Important status

The current directory is a research prototype, not yet an executable end-to-end experiment. Do not use its current outputs as paper results. In particular:

- The Wan2.1-VACE and ProPainter subprocess arguments are assumed by these scripts and have not been adapted/validated against the installed upstream inference entry points.
- `03a_pose_retarget.py` now extracts pose from original frames and source bboxes; however, its pose adapter and Wan input/output contract still need validation against the pinned upstream versions.
- The proposed scale factors (child 0.60, senior 0.95), fixed box aspect ratios, and age prompts are unsupported assumptions. Estimate target geometry from the dataset or specify and validate them as intervention settings.
- A model's apparent-age response is not a behavioral ground truth. The counterfactual only tests sensitivity to rendered age cues, and an image/video editor may also change clothing, body size, gait, occlusion, and local pixels. Human review plus independent pose/track checks are required.
- The current output format expects individual `frame_XXXXXX.png` files. If an upstream model emits a video, implement and validate deterministic frame extraction and alignment before continuing.
- The sanity script has no working V-JEPA 2/Cosmos adapter here. `mock` is for software debugging only and must never be reported as evidence.

Resolve those adapters and validate a single 2–3 second example locally before spending a Kaggle GPU session. Keep the benchmark protocol frozen before comparing methods.

## 1. Prepare Kaggle inputs

Create a Kaggle Notebook with GPU enabled, then attach read-only Kaggle Datasets containing:

1. The repository snapshot (or upload a zip of `counterfactual_pipeline/`).
2. The IDD-PeD video files and a manifest with the required columns documented in `00_select_clips.py`.
3. SAM 2, DWPose, ProPainter, Wan2.1-VACE, and world-model weights only if their licenses permit Kaggle use. Keep large checkpoints in Kaggle Datasets; do not download them in every notebook run.

Use `/kaggle/working` for all generated data. Kaggle input paths are read-only. Start by inspecting GPU, disk, and file paths; do not copy the full video corpus unless you need it.

```python
!nvidia-smi
!df -h /kaggle/working
!find /kaggle/input -maxdepth 3 -type f | head -40
```

## 2. Install and record the environment

Pin exact versions/commits for the four upstream model repositories in the notebook. Install only their required dependencies; a single `pip install -r requirements.txt` does not install the commented-out model dependencies. Record package versions, Git commit hashes, checkpoint identifiers, GPU model, and all generation parameters in a run metadata file. Restart the kernel after dependency changes and verify imports before processing clips.

## 3. Pilot one clip, then freeze the input set

Run clip selection against the manifest and inspect its CSV before generation. Make sure source age, video IDs, frame ranges, annotation quality, and per-age counts are correct. For the stated child-to-adult intervention, analyze child-annotated source clips, with child-to-child sham edits as the primary edit control. Keep adult-to-adult self-edits as a separate sanity control; do not treat adult-to-child outputs as the target intervention.

Run one clip through segmentation and erasure, then visually inspect the source, mask overlay, and erased plate frame by frame. Stop if the target is not tracked, adjacent people are included, a missing mask is treated as background, or the erased region has ghosts. Configure actual image dimensions from extracted frames rather than assuming 1920×1080.

Only after the single-clip pilot passes should you run both candidate editors on the same clip set, same source frames, same target age, and documented seeds/settings. Preserve raw outputs and logs. Never silently continue after a missing checkpoint or failed generation.

## 4. Evaluate without circular metrics

For each output, retain frame alignment and calculate:

- Background preservation outside the union of source and generated-person masks. Use deterministic sampled regions and report missing LPIPS as unavailable, not a pass.
- Independent pose/track footpoint drift. Do not reuse the source footpoint as the edited footpoint; that forces a zero error.
- Age-cue validity, using blinded human ratings and a validated age-perception measure. MiVOLO/CLIP scores are screening signals, not calibrated probabilities or evidence that a person is a child.
- Flicker/temporal consistency and identity consistency across frames.

The current `06_human_validation.py` sheet generator is not adequate for a blinded study: it exposes method/clip information through frame paths, does not bundle viewable videos, and omits source/sham references. Before using it, fix it to export anonymized video assets, preserve the blind key separately, and provide each rating item with its paired source plus sham and target edit. Have at least three raters review randomized, blinded sets. Ask realism, age-cue success, trajectory/pose preservation, background preservation, and artifacts separately. Report agreement with uncertainty. Do not declare a winner from 10 clips or a single aggregate score; report per-clip results and bootstrap confidence intervals, and treat the pilot as method selection only.

## 5. Select a method and produce the final benchmark variants

Pre-register hard quality gates and a tie-break rule before looking at results. Exclude a clip/method pair if any required measure is missing or fails a gate. Among methods that pass, select based on blinded realism and factor-isolation ratings, with artifact and temporal checks as gates. If neither passes, revise the generation method; do not choose the less-bad method and call it validated.

Save the selected method, its full configuration, seeds, source IDs, per-clip QC decisions, and rejected examples. Generate benchmark variants with the frozen method and keep source and edited clips paired. The final analysis should compare model rollouts on paired inputs and report the change attributable to the rendered age intervention, alongside the sham-edit change and editing-quality failures.

## 6. Persist outputs

Write outputs under `/kaggle/working/discretio_run/<run_id>/`. At the end, create a Kaggle Dataset from the notebook output or download the compact artifacts: selected clip manifest, configs, logs, QC CSV/JSON, blinded ratings, and only the needed generated clips. Keep the source videos and large weights in their original licensed datasets.
