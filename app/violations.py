"""Overspeed violation logic."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Set


class ViolationDetector:
    """Flag a vehicle once its speed exceeds the limit for `consecutive` frames in a row."""

    def __init__(self, limit_kmh: float, consecutive: int = 5):
        self.limit = limit_kmh
        self.consecutive = max(1, consecutive)
        self._streak: Dict[int, int] = defaultdict(int)
        self.flagged: Set[int] = set()
        self.first_frame: Dict[int, int] = {}

    def update(self, track_id: int, speed: Optional[float], frame_idx: int) -> bool:
        """Return True only on the frame a vehicle becomes flagged."""
        if speed is None:
            return False
        if speed > self.limit:
            self._streak[track_id] += 1
        else:
            self._streak[track_id] = 0
        if track_id not in self.flagged and self._streak[track_id] >= self.consecutive:
            self.flagged.add(track_id)
            self.first_frame[track_id] = frame_idx
            return True
        return False
