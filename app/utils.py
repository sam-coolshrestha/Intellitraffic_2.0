"""Video IO helpers, browser-playable H.264 re-encoding, colours."""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Tuple

import cv2

log = logging.getLogger("intellitraffic")


def video_info(path: str | Path) -> Tuple[float, int, int, int]:
    """Return (fps, width, height, frame_count). Raises ValueError for unreadable video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if w == 0 or h == 0:
        raise ValueError(f"Video has no readable frames: {path}")
    return float(fps), w, h, n


def make_writer(path: str | Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {path}")
    return writer


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - fall back to system ffmpeg
        return "ffmpeg"


def reencode_h264(src: str | Path, dst: str | Path) -> bool:
    """Re-encode to H.264/yuv420p with faststart so st.video() plays it. Returns success."""
    cmd = [
        _ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-crf", "23", "-movflags", "+faststart", "-an", str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return Path(dst).exists() and Path(dst).stat().st_size > 0
    except Exception as exc:
        log.warning("H.264 re-encode failed: %s", exc)
        return False


def is_h264(path: str | Path) -> bool:
    """Check the first video stream codec by parsing `ffmpeg -i` output."""
    res = subprocess.run([_ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True)
    return bool(re.search(r"Video:\s*h264", res.stderr))


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Stable BGR colour per track id."""
    import colorsys

    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)
