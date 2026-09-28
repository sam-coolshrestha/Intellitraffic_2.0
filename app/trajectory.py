"""Trajectory storage, fading-trail drawing and heatmap generation."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd

from .utils import color_for_id


class TrajectoryStore:
    def __init__(self) -> None:
        self.points: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

    def add(self, track_id: int, frame_idx: int, x: float, y: float) -> None:
        self.points[track_id].append((frame_idx, x, y))

    def draw(self, frame: np.ndarray, track_ids: Iterable[int], trail_length: int = 30) -> None:
        """Draw fading trails in place for the given ids."""
        for tid in track_ids:
            pts = self.points.get(tid, [])[-trail_length:]
            color = color_for_id(tid)
            n = len(pts)
            for i in range(1, n):
                alpha = i / n
                c = tuple(int(v * (0.3 + 0.7 * alpha)) for v in color)
                p1 = (int(pts[i - 1][1]), int(pts[i - 1][2]))
                p2 = (int(pts[i][1]), int(pts[i][2]))
                cv2.line(frame, p1, p2, c, max(1, int(1 + 2 * alpha)), cv2.LINE_AA)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [(tid, f, x, y) for tid, pts in self.points.items() for f, x, y in pts]
        return pd.DataFrame(rows, columns=["vehicle_id", "frame", "x", "y"])

    def heatmap(self, background: np.ndarray) -> np.ndarray:
        """Overlay accumulated centroid density on a background frame (BGR)."""
        h, w = background.shape[:2]
        acc = np.zeros((h, w), dtype=np.float32)
        for pts in self.points.values():
            for _, x, y in pts:
                xi, yi = int(x), int(y)
                if 0 <= xi < w and 0 <= yi < h:
                    acc[yi, xi] += 1.0
        if acc.max() == 0:
            return background.copy()
        acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=max(5, w // 80))
        acc = (255 * acc / acc.max()).astype(np.uint8)
        colored = cv2.applyColorMap(acc, cv2.COLORMAP_JET)
        mask = (acc > 8)[..., None]
        blended = cv2.addWeighted(background, 0.5, colored, 0.5, 0)
        return np.where(mask, blended, background)
