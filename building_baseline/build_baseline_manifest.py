#!/usr/bin/env python3
"""
build_baseline_manifest.py

Constructs the master baseline event manifest and Distributional Ground Truth
lookup table E from IDD-PeD annotations for the DISCRETIO benchmark.

Pipeline Order:
  1. Ingest XML annotations (pedestrian tracks) and POI metadata.
  2. Load vehicle OBD telemetry (speed in km/h → convert to m/s).
  3. Detect anchor frame t0 per crossing episode.
  4. Slice context [t0-T_CTX, t0] and outcome [t0+1, t0+T_OUT] windows.
  5. Compute kinematics (ground-plane projection, ego-compensated).
  6. Compute bucket keys at t0.
  7. Label outcomes (cross, jaywalk, wait, censored_truncated) + hesitation count.
  8. Export manifest CSV + Parquet.

Verified against IDD-PeD gp_set_0001_vid_0001.xml and _obd.xml.
"""

import os
import glob
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# ==============================================================================
# TEMPORAL / WINDOW PARAMETERS
# IDD-PeD GoPro videos are 30 fps (native). Confirmed from crossing_point gaps.
# Context: 60 frames = 2.0 s  (satisfies Cosmos/Genie/V-JEPA2/BiTraP minimum)
# Outcome: 90 frames = 3.0 s
# Min gap between episodes from the same track: 150 frames = 5.0 s
# ==============================================================================
FPS = 30.0
T_CTX_FRAMES = 60     # Frames of context BEFORE t0 (inclusive)
T_OUT_FRAMES = 90     # Frames of outcome AFTER t0 (inclusive)
MIN_EPISODE_GAP = 150 # Min frame gap between two events from same pedestrian
MIN_TRACK_FRAMES = T_CTX_FRAMES + 10  # Reject tracks shorter than this

# ==============================================================================
# CAMERA INTRINSICS (GoPro Hero 7/8, 1080p Wide, dashcam mount)
# These are approximate; replace with calibrated values if available.
# ==============================================================================
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
CAMERA_HEIGHT_M = 1.35          # Dashcam mounting height on windshield
CAMERA_FOCAL_LENGTH_PX = 1050.0 # Approx for 118-deg diagonal WFOV at 1920px
OPTICAL_CENTER = (IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0)


# ==============================================================================
# ANNOTATION LABEL MAPS (from _map_text_to_scalar in iddped_interface_traj.py)
# ==============================================================================

# CB values that mean the pedestrian IS crossing
CB_CROSSING = {"CU", "CFU", "CD", "CFD"}
# CB value that means crossing was interrupted / hesitation
CB_HESITATE = {"CI"}

# TI (Traffic Interaction) values that signal active interaction / hesitation
TI_INTERACTION = {"WTT", "HG"}   # WTT = Weaving Through Traffic, HG = Hand Gesture

# signalized_type values that mean there IS a signal
SIGNALIZED_VALUES = {"S", "CS"}  # S = Signal only, CS = Crosswalk + Signal


# ==============================================================================
# DATA SCHEMA
# ==============================================================================
@dataclass
class BaselineEventRecord:
    # Identifiers
    event_id: str
    video_id: str
    track_id: str
    cluster_video_id: str     # = set_id, e.g. gp_set_0001; used for cluster bootstrap

    # Temporal Windows (all in absolute video frame numbers)
    start_frame: int          # = t0 - T_CTX_FRAMES
    t0_frame: int             # Anchor decision frame
    end_frame: int            # = t0 + T_OUT_FRAMES
    t0_timestamp_s: float     # t0 in seconds

    # Bounding Box at t0
    xtl_t0: float
    ytl_t0: float
    xbr_t0: float
    ybr_t0: float

    # Occlusion state at t0
    # Source: child attribute 'occlusion' ("None","Part","Full"),
    # cross-checked against box-level 'occluded' CVAT flag (0/1).
    occlusion_t0: str         # "none", "partial", "full"

    # Bucket Keys (computed at t0)
    age_bracket: str          # "child", "adult", "senior"
    gender: str               # "male", "female", "unknown"
    ego_speed_mps: float      # From OBD_speed (km/h ÷ 3.6 → m/s)
    ped_speed_mps: float      # Estimated from bounding-box displacement
    ttc_s: float              # Time-to-Collision (ground-plane, ego-compensated)
    signal_type: str          # "signalized" or "unsignalized"
                              #   primary: POI signalized_type in {S, CS}
                              #   fallback: ped box 'near signal' == 'yes'
    road_type: str            # "main","secondary","street","lane"
    interaction_flag: int     # 1 if TI∈{WTT,HG} or Joint Interaction≠N/A at t0

    # Composite bucket hash for lookup table E
    bucket_hash: str          # e.g. "adult_medium_approaching_unsignalized_int0"

    # Outcome Labels (over [t0+1, t0+T_OUT])
    outcome: str              # "cross","jaywalk","wait","censored_truncated"
    hesitation_count: int     # Distinct hesitation events in outcome window
    is_truncated: bool        # Track left camera view before outcome window ends
    is_solo: bool             # No other pedestrian within 3m (group_size == 1)
    crossing_motive: float    # 1.0=intentional, 0.5=maybe, 0.0=no (from POI)


# ==============================================================================
# OBD TELEMETRY LOADER
# ==============================================================================
def load_obd(obd_path: str) -> Dict[int, float]:
    """
    Parses *_obd.xml and returns {frame_id: ego_speed_mps}.
    OBD_speed is in km/h → divide by 3.6 for m/s.
    Frame ids in OBD are 0-indexed and match annotation frame numbers.
    """
    if not os.path.isfile(obd_path):
        return {}
    try:
        root = ET.parse(obd_path).getroot()
        return {
            int(f.get("id")): float(f.get("OBD_speed", 0.0)) / 3.6
            for f in root.findall("./frame")
        }
    except Exception as e:
        print(f"  [WARN] Could not parse OBD {obd_path}: {e}")
        return {}


# ==============================================================================
# KINEMATICS
# ==============================================================================
def project_box_to_ground(box: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Flat-road ground-plane projection from monocular bounding box bottom-centre.
    Returns [X_lat, Z_depth] in metres (camera coordinate frame).

    Z = f·h / (y_br - v0),   X = (u_bc - u0)·Z / f
    Clamped to avoid singularity near the horizon line.
    """
    xtl, ytl, xbr, ybr = box
    u_bc = (xtl + xbr) / 2.0
    v_br = ybr
    u0, v0 = OPTICAL_CENTER
    f = CAMERA_FOCAL_LENGTH_PX
    h = CAMERA_HEIGHT_M

    denom = max(v_br - v0, 5.0)   # Never divide by zero near horizon
    Z = (f * h) / denom
    X = (u_bc - u0) * Z / f
    return np.array([X, Z])


def compute_kinematics(
    boxes_at_frames: Dict[int, Tuple[float, float, float, float]],
    t0: int,
    obd: Dict[int, float],
) -> Tuple[float, float, float]:
    """
    Returns (ped_speed_mps, ped_heading_deg, ttc_s) at frame t0.
    Uses up to 3 earlier frames for finite-difference velocity.
    Subtracts ego forward velocity for TTC and speed calculation.
    """
    dt = 1.0 / FPS
    ego_speed = obd.get(t0, obd.get(t0 - 1, obd.get(t0 + 1, 5.56)))

    pos_curr = project_box_to_ground(boxes_at_frames[t0])

    # Find nearest earlier annotated frame (may not be t0-1 due to keyframe gaps)
    prev_frames = sorted([f for f in boxes_at_frames if f < t0])
    if prev_frames:
        f_prev = prev_frames[-1]
        gap_frames = t0 - f_prev
        if gap_frames <= 5:   # Only use if nearby (≤ 5 frames)
            pos_prev = project_box_to_ground(boxes_at_frames[f_prev])
            actual_dt = gap_frames * dt
            d_cam = (pos_curr - pos_prev) / actual_dt
        else:
            d_cam = np.array([0.0, 0.0])
    else:
        d_cam = np.array([0.0, 0.0])

    # Pedestrian world velocity (ego-motion compensated along Z)
    d_world = d_cam.copy()
    d_world[1] += ego_speed  # Camera Z faces forward; car moves in +Z

    ped_speed = float(np.linalg.norm(d_world))
    ped_heading = float(np.degrees(np.arctan2(d_world[0], d_world[1])) % 360.0)

    # TTC: how long until the pedestrian crosses the vehicle path at Z=0?
    # closing_speed = ego longitudinal component + pedestrian closing component
    closing = ego_speed - d_cam[1]   # relative closing speed along Z
    if closing > 0.3:
        ttc = float(pos_curr[1] / closing)
    else:
        ttc = 99.0    # Diverging or stationary relative trajectory

    return round(ped_speed, 3), round(ped_heading, 1), round(max(ttc, 0.1), 2)


# ==============================================================================
# ANCHOR FRAME SELECTION
# Deterministic rule (applied identically to every track):
#   R1 (primary): POI crossing_point if it is a valid video frame (>0) AND
#                 that frame exists in the pedestrian track.
#   R2 (fallback for crossing=yes): first frame where CB ∈ {CU,CFU,CD,CFD,CI}.
#   R3 (fallback for crossing=no):  first frame where TTC ≤ 5.0 s (interaction horizon).
#   R4 (last resort):               frame at index T_CTX_FRAMES in sorted track.
#
# After selection, enforce T_CTX_FRAMES prior frames exist; shift t0 forward if needed.
# ==============================================================================
def pick_anchor_t0(
    track_frames: List[int],
    boxes: Dict[int, Tuple[float, float, float, float]],
    frame_attrs: Dict[int, Dict[str, str]],
    poi_attrs: Dict[str, str],
    obd: Dict[int, float],
) -> Optional[int]:

    if len(track_frames) < MIN_TRACK_FRAMES:
        return None

    crossing_point_raw = poi_attrs.get("crossing_point", "-10")
    try:
        cp_frame = int(crossing_point_raw)
    except (ValueError, TypeError):
        cp_frame = -10

    candidate = None

    # R1: annotated crossing_point (ignore sentinel -10)
    if cp_frame > 0 and cp_frame in boxes:
        candidate = cp_frame

    # R2: onset of CB crossing tag
    if candidate is None:
        for f in track_frames:
            cb = frame_attrs.get(f, {}).get("CB", "")
            if cb in CB_CROSSING | CB_HESITATE:
                candidate = f
                break

    # R3: first frame within TTC interaction horizon
    if candidate is None:
        for f in track_frames:
            ego = obd.get(f, 5.56)
            _, _, ttc = compute_kinematics(boxes, f, obd)
            if ttc <= 5.0:
                candidate = f
                break

    # R4: positional fallback
    if candidate is None:
        if len(track_frames) >= (T_CTX_FRAMES + T_OUT_FRAMES):
            candidate = track_frames[T_CTX_FRAMES]
        else:
            return None

    # Enforce context window: enough prior frames must exist
    idx = track_frames.index(candidate)
    if idx < T_CTX_FRAMES:
        # Shift t0 forward until T_CTX_FRAMES prior exist
        if len(track_frames) > T_CTX_FRAMES:
            candidate = track_frames[T_CTX_FRAMES]
        else:
            return None

    # Final check: outcome window must also have at least some frames
    outcome_frames_available = [f for f in track_frames if f > candidate]
    if len(outcome_frames_available) < 5:
        return None

    return candidate


# ==============================================================================
# OUTCOME LABELING
# ==============================================================================
def label_outcome(
    track_frames: List[int],
    boxes: Dict[int, Tuple[float, float, float, float]],
    frame_attrs: Dict[int, Dict[str, str]],
    poi_attrs: Dict[str, str],
    t0: int,
    obd: Dict[int, float],
) -> Tuple[str, int, bool]:
    """
    Returns (outcome, hesitation_count, is_truncated) over [t0+1, t0+T_OUT].

    outcome is one of:
      "cross"             — pedestrian traverses the vehicle path
      "jaywalk"           — crosses without crosswalk or against signal
      "wait"              — does not enter the road corridor
      "censored_truncated"— track exits frame before window ends

    Pitfall: track exiting frame ≠ "did not cross"; these are excluded from E.
    """
    outcome_frames = sorted([f for f in track_frames if t0 < f <= (t0 + T_OUT_FRAMES)])

    # --- Truncation check ---
    # Truncated if track ends before 70% of the outcome window is covered
    # AND the last known bounding box exits a screen-edge margin.
    is_truncated = False
    if len(outcome_frames) < int(T_OUT_FRAMES * 0.7):
        if outcome_frames:
            last_box = boxes[outcome_frames[-1]]
            xtl, ytl, xbr, ybr = last_box
            at_left_edge  = xtl <= 30
            at_right_edge = xbr >= (IMAGE_WIDTH - 30)
            at_top_edge   = ytl <= 30
            # Near-bottom exit is ambiguous (can be very close pedestrian)
            if at_left_edge or at_right_edge or at_top_edge:
                is_truncated = True
        else:
            # No outcome frames at all → definitely truncated
            is_truncated = True

    if is_truncated:
        return "censored_truncated", 0, True

    # --- Hesitation counting ---
    # Count: (a) CI/HG/WTT occurrences and (b) discrete stop→start speed transitions
    hesitation_count = 0
    in_pause = False
    for f in outcome_frames:
        cb = frame_attrs.get(f, {}).get("CB", "")
        ti = frame_attrs.get(f, {}).get("TI", "")

        if cb in CB_HESITATE or ti in TI_INTERACTION:
            hesitation_count += 1

        if f in boxes:
            v, _, _ = compute_kinematics(boxes, f, obd)
            if v < 0.3 and not in_pause:
                in_pause = True
                hesitation_count += 1
            elif v >= 0.6 and in_pause:
                in_pause = False

    # --- Outcome classification (priority order) ---
    # Use POI-level labels as primary source (annotated at track level).
    # Per-frame CB in the outcome window as secondary evidence.
    poi_crossing  = (poi_attrs.get("crossing", "no").lower() == "yes")
    poi_category  = poi_attrs.get("category", "default").lower()

    # Check if crosswalk was used (crosswalk_usage in {"no","N/A"} means no crosswalk)
    crosswalk_usage = poi_attrs.get("crosswalk_usage", "N/A").lower()
    no_crosswalk = crosswalk_usage in {"no", "n/a"}

    # Active crossing in outcome window
    cb_in_outcome = {frame_attrs.get(f, {}).get("CB", "") for f in outcome_frames}
    active_crossing = bool(cb_in_outcome & CB_CROSSING)

    if poi_crossing or active_crossing or poi_category in {"crossing", "jaywalking"}:
        if poi_category == "jaywalking" or no_crosswalk:
            outcome = "jaywalk"
        else:
            outcome = "cross"
    else:
        outcome = "wait"

    return outcome, hesitation_count, False


# ==============================================================================
# CORE PARSER
# ==============================================================================
def parse_video_xml(
    xml_path: str,
    obd: Dict[int, float],
) -> List[BaselineEventRecord]:
    """
    Parses one IDD-PeD video annotation XML and returns a list of
    BaselineEventRecord entries — one per valid crossing episode.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as e:
        print(f"  [SKIP] XML parse error in {os.path.basename(xml_path)}: {e}")
        return []

    # Validate it's an IDD-PeD file (must have both POI and pedestrian tracks)
    has_poi = len(root.findall('./track[@label="POI"]')) > 0
    has_ped = len(root.findall('./track[@label="pedestrian"]')) > 0
    if not (has_poi and has_ped):
        return []

    video_id = os.path.splitext(os.path.basename(xml_path))[0]
    # cluster_video_id = set identifier, e.g. gp_set_0001 from gp_set_0001_vid_0001
    cluster_video_id = "_".join(video_id.split("_")[:3]) if "_vid_" in video_id else video_id

    # ---- Build POI lookup: pid → attribute dict ----
    # POI tracks have exactly 2 boxes (first and second keyframe).
    # All high-level attributes live on every box; read from box[0].
    poi_map: Dict[str, Dict[str, str]] = {}
    for t in root.findall('./track[@label="POI"]'):
        boxes = t.findall("./box")
        if not boxes:
            continue
        attrs = {a.get("name"): (a.text or "") for a in boxes[0].findall("./attribute")}
        pid = attrs.get("id")
        if pid:
            poi_map[pid] = attrs

    # ---- Build pedestrian track data ----
    records: List[BaselineEventRecord] = []

    for t in root.findall('./track[@label="pedestrian"]'):
        raw_boxes = t.findall("./box")
        if not raw_boxes:
            continue

        # The pedestrian id lives as a child attribute element inside box[0].
        id_attr = raw_boxes[0].find('./attribute[@name="id"]')
        if id_attr is None or not id_attr.text:
            continue
        ped_id = id_attr.text

        poi_attrs = poi_map.get(ped_id, {})

        # Collect valid (non-outside) box data
        track_frames: List[int] = []
        boxes_dict: Dict[int, Tuple[float, float, float, float]] = {}
        frame_attrs: Dict[int, Dict[str, str]] = {}

        for b in raw_boxes:
            if b.get("outside") == "1":
                continue
            frame = int(b.get("frame"))
            box = (
                float(b.get("xtl")),
                float(b.get("ytl")),
                float(b.get("xbr")),
                float(b.get("ybr")),
            )
            # Frame-level attributes: use short attribute name keys (CB, TI, AI, etc.)
            # The child attribute 'occlusion' ("None","Part","Full") is the fine-grained
            # occlusion state; the CVAT box attrib 'occluded' is a binary flag (0/1).
            b_attrs = {a.get("name"): (a.text or "") for a in b.findall("./attribute")}
            b_attrs["_occluded_flag"] = b.get("occluded", "0")   # keep CVAT flag separate

            track_frames.append(frame)
            boxes_dict[frame] = box
            frame_attrs[frame] = b_attrs

        if len(track_frames) < MIN_TRACK_FRAMES:
            continue

        track_frames.sort()

        # ---- Select anchor frame t0 ----
        t0 = pick_anchor_t0(track_frames, boxes_dict, frame_attrs, poi_attrs, obd)
        if t0 is None:
            continue

        # ---- Kinematics at t0 ----
        ped_speed, ped_heading, ttc = compute_kinematics(boxes_dict, t0, obd)
        ego_speed = obd.get(t0, obd.get(t0 - 1, obd.get(t0 + 1, 5.56)))

        # ---- Demographic attributes ----
        raw_age = poi_attrs.get("age", "adult").strip().lower()
        age_bracket = (
            "child" if raw_age == "child"
            else "senior" if raw_age == "senior"
            else "adult"   # covers "adult" and "teenager"
        )
        raw_gender = poi_attrs.get("gender", "default").strip().lower()
        gender = raw_gender if raw_gender in {"male", "female"} else "unknown"

        # ---- Signal type ----
        # Primary: POI-level signalized_type annotation
        sig_type_raw = poi_attrs.get("signalized_type", "N/A").strip()
        if sig_type_raw in SIGNALIZED_VALUES:
            signal_type = "signalized"
        else:
            # Fallback: per-frame 'near signal' attribute at t0
            near_sig = frame_attrs.get(t0, {}).get("near signal", "no").lower()
            signal_type = "signalized" if near_sig == "yes" else "unsignalized"

        road_type = poi_attrs.get("road_type", "main").strip()

        # ---- Interaction flag at t0 ----
        t0_attrs = frame_attrs.get(t0, {})
        ti_t0 = t0_attrs.get("TI", "")
        ji_t0 = t0_attrs.get("Joint Interaction", "N/A")
        interaction_flag = 1 if (ti_t0 in TI_INTERACTION or ji_t0 not in {"N/A", ""}) else 0

        # ---- Bounding box & occlusion at t0 ----
        xtl_t0, ytl_t0, xbr_t0, ybr_t0 = boxes_dict[t0]
        # Use the annotated child attribute 'occlusion' (more precise than binary flag)
        occ_child = t0_attrs.get("occlusion", "").strip()
        if occ_child.lower() in {"full"}:
            occlusion_t0 = "full"
        elif occ_child.lower() in {"part", "partial"}:
            occlusion_t0 = "partial"
        elif t0_attrs.get("_occluded_flag") == "1":
            # CVAT binary flag says occluded, but child attribute is missing/None
            occlusion_t0 = "partial"
        else:
            occlusion_t0 = "none"

        # ---- Bucket hash ----
        # Speed bins based on EGO speed (what the model observes as scene dynamics)
        if ego_speed < 2.78:         # < 10 km/h
            speed_bin = "slow"
        elif ego_speed < 8.33:       # 10–30 km/h
            speed_bin = "medium"
        else:                        # > 30 km/h
            speed_bin = "fast"

        if ttc < 2.5:
            ttc_bin = "urgent"
        elif ttc < 4.5:
            ttc_bin = "approaching"
        else:
            ttc_bin = "distant"

        bucket_hash = f"{age_bracket}_{speed_bin}_{ttc_bin}_{signal_type}_int{interaction_flag}"

        # ---- Outcome labeling ----
        outcome, hesitation_count, is_truncated = label_outcome(
            track_frames, boxes_dict, frame_attrs, poi_attrs, t0, obd
        )

        # ---- Solo pedestrian flag (group_size == 1) ----
        try:
            group_size = int(poi_attrs.get("group_size", "1"))
        except ValueError:
            group_size = 1
        is_solo = (group_size == 1)

        # crossing_motive
        try:
            crossing_motive_val = float(poi_attrs.get("crossing_motive", "0") or "0")
        except ValueError:
            crossing_motive_val = 0.0

        event_id = f"{video_id}__{ped_id}__t0{t0}"
        records.append(BaselineEventRecord(
            event_id=event_id,
            video_id=video_id,
            track_id=ped_id,
            cluster_video_id=cluster_video_id,
            start_frame=t0 - T_CTX_FRAMES,
            t0_frame=t0,
            end_frame=t0 + T_OUT_FRAMES,
            t0_timestamp_s=round(t0 / FPS, 2),
            xtl_t0=round(xtl_t0, 2),
            ytl_t0=round(ytl_t0, 2),
            xbr_t0=round(xbr_t0, 2),
            ybr_t0=round(ybr_t0, 2),
            occlusion_t0=occlusion_t0,
            age_bracket=age_bracket,
            gender=gender,
            ego_speed_mps=round(ego_speed, 3),
            ped_speed_mps=round(ped_speed, 3),
            ttc_s=ttc,
            signal_type=signal_type,
            road_type=road_type,
            interaction_flag=interaction_flag,
            bucket_hash=bucket_hash,
            outcome=outcome,
            hesitation_count=hesitation_count,
            is_truncated=is_truncated,
            is_solo=is_solo,
            crossing_motive=crossing_motive_val,
        ))

    return records


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build DISCRETIO master baseline event manifest from IDD-PeD."
    )
    parser.add_argument(
        "--xml_dir", type=str, default="sample_data:IDD-Ped",
        help="Root directory containing IDD-PeD annotation XML files (searched recursively)."
    )
    parser.add_argument(
        "--obd_dir", type=str, default=None,
        help="Root directory for OBD telemetry XML files. If None, a zero-speed fallback is used."
    )
    parser.add_argument("--output_csv",     type=str, default="idd_ped_master_manifest.csv")
    parser.add_argument("--output_parquet", type=str, default="idd_ped_master_manifest.parquet")
    args = parser.parse_args()

    # Discover annotation XMLs (skip OBD files)
    xml_files = glob.glob(os.path.join(args.xml_dir, "**", "*.xml"), recursive=True)
    xml_files += glob.glob(os.path.join(args.xml_dir, "*.xml"))
    xml_files = sorted(set(f for f in xml_files if "obd" not in os.path.basename(f).lower()))
    print(f"Found {len(xml_files)} annotation XML file(s).")

    all_records: List[BaselineEventRecord] = []

    for xml_path in xml_files:
        video_id = os.path.splitext(os.path.basename(xml_path))[0]
        print(f"  Processing: {video_id}")

        # Load OBD telemetry for this video if available
        obd: Dict[int, float] = {}
        if args.obd_dir:
            # Convention: annotations_vehicle/gopro/<set>/<video>_obd.xml
            obd_candidate = os.path.join(args.obd_dir, f"{video_id}_obd.xml")
            obd = load_obd(obd_candidate)
            if not obd:
                # Walk the obd_dir to find a matching file
                for obd_path in glob.glob(os.path.join(args.obd_dir, "**", f"{video_id}_obd.xml"), recursive=True):
                    obd = load_obd(obd_path)
                    if obd:
                        break
            if obd:
                print(f"    OBD: {len(obd)} frames loaded.")
            else:
                print(f"    OBD: not found, using fallback 5.56 m/s (20 km/h).")

        records = parse_video_xml(xml_path, obd)
        print(f"    Events extracted: {len(records)}")
        all_records.extend(records)

    if not all_records:
        print("\nNo events extracted. Verify that --xml_dir points to IDD-PeD annotation XMLs.")
        return

    df = pd.DataFrame([asdict(r) for r in all_records])

    # ---- Export ----
    df.to_csv(args.output_csv, index=False)
    df.to_parquet(args.output_parquet, index=False)
    print(f"\nManifest saved: {args.output_csv}  ({args.output_parquet})")

    # ---- Summary Statistics ----
    valid = df[df["outcome"] != "censored_truncated"]
    print("\n" + "=" * 62)
    print("MASTER MANIFEST STATISTICS (LOOKUP TABLE E BASELINE)")
    print("=" * 62)
    print(f"  Total candidate events   : {len(df)}")
    print(f"  Valid (non-censored)     : {len(valid)}")
    print(f"  Truncated / censored     : {(df['is_truncated']).sum()}")
    print(f"  Unique videos            : {df['video_id'].nunique()}")
    print(f"  Unique set clusters      : {df['cluster_video_id'].nunique()}")
    print(f"\n  Age distribution:")
    print(df["age_bracket"].value_counts().to_string(header=False))
    print(f"\n  Outcome distribution (valid events):")
    print(valid["outcome"].value_counts().to_string(header=False))
    print(f"\n  Signal type:")
    print(df["signal_type"].value_counts().to_string(header=False))
    print(f"\n  Mean hesitation count    : {valid['hesitation_count'].mean():.2f}")
    print(f"\n  Bucket coverage (top 10):")
    print(valid["bucket_hash"].value_counts().head(10).to_string(header=False))


if __name__ == "__main__":
    main()
