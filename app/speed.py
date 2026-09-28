"""Speed estimation with simple (m/px) or homography (image->road plane) calibration."""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from .config import SpeedCfg


class SpeedEstimator:
    def __init__(self, cfg: SpeedCfg, fps: float):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.cfg = cfg
        self.fps = fps
        self._hist: Dict[int, Deque[Tuple[int, float, float]]] = defaultdict(lambda: deque(maxlen=max(2, cfg.window)))
        self._smooth: Dict[int, float] = {}
        self._H: Optional[np.ndarray] = None
        if cfg.mode == "homography":
            src = np.array(cfg.image_points, dtype=np.float32)
            dst = np.array(cfg.world_points, dtype=np.float32)
            if src.shape != (4, 2) or dst.shape != (4, 2):
                raise ValueError("Homography mode needs exactly 4 image points and 4 world points ([x, y] each).")
            self._H = cv2.getPerspectiveTransform(src, dst)
        elif cfg.mode != "simple":
            raise ValueError(f"Unknown speed mode: {cfg.mode}")
        elif cfg.meters_per_pixel <= 0:
            raise ValueError("meters_per_pixel must be positive")

    def to_world(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """Map an image point (pixels) to metres."""
        if self._H is None:
            return pt[0] * self.cfg.meters_per_pixel, pt[1] * self.cfg.meters_per_pixel
        out = cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), self._H)[0][0]
        return float(out[0]), float(out[1])

    def update(self, track_id: int, frame_idx: int, pt: Tuple[float, float]) -> Optional[float]:
        """Add an observation; return smoothed km/h or None while warming up."""
        x, y = self.to_world(pt)
        h = self._hist[track_id]
        h.append((frame_idx, x, y))
        if len(h) < self.cfg.min_frames:
            return None
        f0, x0, y0 = h[0]
        f1, x1, y1 = h[-1]
        dt = (f1 - f0) / self.fps
        if dt <= 0:
            return self._smooth.get(track_id)
        raw = math.hypot(x1 - x0, y1 - y0) / dt * 3.6
        prev = self._smooth.get(track_id)
        a = self.cfg.smoothing
        val = raw if prev is None else a * raw + (1 - a) * prev
        self._smooth[track_id] = val
        return val
