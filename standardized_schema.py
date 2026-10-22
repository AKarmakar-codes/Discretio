#!/usr/bin/env python3
"""
standardized_schema.py
The core data contract and schema definitions for the DISCRETIO Distributional Ground Truth.
All team members (Person 1, Person 2, Person 3) must format their extracted tracks
into this standard schema before passing to the bucketing engine.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np


class AgeBracket(str, Enum):
    CHILD = "child"
    ADULT = "adult"
    SENIOR = "senior"


class BehaviorOutcome(str, Enum):
    CROSS = "cross"
    HESITATE = "hesitate"
    STOP = "stop"


class CrossingContext(str, Enum):
    SIGNALIZED_CROSSWALK = "signalized_crosswalk"
    UNSIGNALIZED_CROSSWALK = "unsignalized_crosswalk"
    INFORMAL_CROSSING = "informal_crossing"


class Environment(str, Enum):
    UNSTRUCTURED_INDIA = "unstructured_india"
    STRUCTURED_WESTERN = "structured_western"
    STRUCTURED_ASIAN = "structured_asian"


@dataclass
class StandardizedTrackPoint:
    dataset: str                      # "IDD-PeD", "JAAD", "PIE", "TITAN"
    clip_id: str                      # Video sequence or clip identifier
    track_id: str                     # Unique pedestrian identifier within clip
    frame_id: int                     # Video frame number (0-indexed)
    timestamp_s: float                # Elapsed time in seconds
    bbox_2d: Tuple[float, float, float, float]  # (xtl, ytl, xbr, ybr)
    occlusion: str                    # "none", "partial", "full"
    age_bracket: str                  # "child", "adult", "senior"
    gender: str                       # "male", "female", "unknown"
    ped_speed_mps: float              # Pedestrian estimated speed (m/s)
    ped_heading_deg: float            # Pedestrian heading (0-360 deg)
    ego_speed_mps: float              # Vehicle ego speed (m/s)
    ttc_s: float                      # Time-to-Collision to crossing path (s)
    crossing_context: str             # "signalized_crosswalk", "unsignalized_crosswalk", "informal_crossing"
    environment: str                  # "unstructured_india", "structured_western", "structured_asian"
    outcome: str                      # "cross", "hesitate", "stop"

    def to_dict(self):
        d = asdict(self)
        d["xtl"], d["ytl"], d["xbr"], d["ybr"] = self.bbox_2d
        del d["bbox_2d"]
        return d


def compute_monocular_kinematics(
    bbox_prev: Tuple[float, float, float, float],
    bbox_curr: Tuple[float, float, float, float],
    dt: float,
    ego_speed_mps: float = 0.0,
    camera_focal_length_px: float = 1000.0,
    camera_height_m: float = 1.35,
    camera_optical_center: Tuple[float, float] = (960.0, 540.0)
) -> Tuple[float, float, float]:
    """
    Estimates pedestrian speed (m/s), heading (degrees), and TTC (seconds)
    from consecutive monocular 2D bounding boxes using flat-ground road plane assumptions.
    """
    u0, v0 = camera_optical_center
    f = camera_focal_length_px
    h = camera_height_m

    def get_ground_pos(box):
        xtl, ytl, xbr, ybr = box
        ub = (xtl + xbr) / 2.0
        vb = ybr
        # Avoid division by zero when near horizon
        denom = max(vb - v0, 5.0)
        z = (f * h) / denom
        x = (ub - u0) * z / f
        return np.array([x, z])

    pos_prev = get_ground_pos(bbox_prev)
    pos_curr = get_ground_pos(bbox_curr)

    # Relative displacement vector in camera coordinates
    d_rel = (pos_curr - pos_prev) / max(dt, 1e-4)

    # Adjust for ego-motion (assuming ego moves along camera Z axis)
    d_ped = d_rel.copy()
    d_ped[1] += ego_speed_mps

    speed_mps = float(np.linalg.norm(d_ped))
    heading_deg = float(np.degrees(np.arctan2(d_ped[0], d_ped[1])) % 360.0)

    # Time-To-Collision: current longitudinal distance / relative closing speed
    closing_speed = ego_speed_mps - d_rel[1]
    if closing_speed > 0.5:
        ttc_s = float(pos_curr[1] / closing_speed)
    else:
        ttc_s = 99.0  # Safe / non-converging

    return round(speed_mps, 2), round(heading_deg, 1), round(max(ttc_s, 0.1), 2)


if __name__ == "__main__":
    print("Standardized schema initialized successfully.")
    # Quick sanity test of kinematics calculation
    box0 = (900.0, 550.0, 1020.0, 750.0)
    box1 = (905.0, 552.0, 1025.0, 755.0)
    v, th, ttc = compute_monocular_kinematics(box0, box1, dt=0.1, ego_speed_mps=8.33)
    print(f"Sanity Check: Speed = {v} m/s, Heading = {th} deg, TTC = {ttc} s")

