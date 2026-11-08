"""
utils/video_io.py
-----------------
Frame extraction and clip reassembly utilities for the DISCRETIO
counterfactual pipeline.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str,
    out_dir: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    fps: float = 30.0,
) -> int:
    """
    Extract frames [start_frame, end_frame] from a video into out_dir as
    zero-padded PNGs: frame_000000.png, frame_000001.png, ...

    Returns the number of frames written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    count = 0
    frame_idx = start_frame
    while True:
        if end_frame is not None and frame_idx > end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        out_path = out_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(out_path), frame)
        count += 1
        frame_idx += 1

    cap.release()
    return count


def frames_to_array(frames_dir: str, pattern: str = "frame_*.png") -> np.ndarray:
    """
    Load a sorted sequence of PNG frames from frames_dir into a uint8 array
    of shape (T, H, W, 3) in RGB order.
    """
    from glob import glob

    paths = sorted(glob(os.path.join(frames_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No frames matching {pattern} in {frames_dir}")

    frames = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise IOError(f"Could not read frame: {p}")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return np.stack(frames, axis=0)


def array_to_frames(
    frames: np.ndarray,
    out_dir: str,
    start_index: int = 0,
    rgb: bool = True,
) -> None:
    """
    Write a (T, H, W, 3) uint8 RGB array to out_dir as zero-padded PNGs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if rgb else frame
        cv2.imwrite(str(out_dir / f"frame_{start_index + i:06d}.png"), img)


# ---------------------------------------------------------------------------
# Clip assembly
# ---------------------------------------------------------------------------

def frames_to_video(
    frames_dir: str,
    output_path: str,
    fps: float = 30.0,
    crf: int = 18,
    pattern: str = "frame_%06d.png",
) -> None:
    """
    Re-assemble PNG frames in frames_dir into an MP4 using ffmpeg.
    Uses libx264 with the given CRF (lower = higher quality).
    """
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, pattern),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def get_video_info(video_path: str) -> Tuple[int, int, float, int]:
    """
    Returns (width, height, fps, total_frames) for a video file.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, n
