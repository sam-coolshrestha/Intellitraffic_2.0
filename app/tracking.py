"""Per-vehicle track history (ByteTrack itself runs inside ultralytics)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .detection import Detection


@dataclass
class TrackRecord:
    track_id: int
    first_frame: int
    last_frame: int
    frames_seen: int = 0
    class_votes: Counter = field(default_factory=Counter)
    centroids: List[Tuple[int, float, float]] = field(default_factory=list)  # frame, x, y

    @property
    def cls(self) -> str:
        return self.class_votes.most_common(1)[0][0] if self.class_votes else "unknown"


class TrackHistory:
    def __init__(self) -> None:
        self.tracks: Dict[int, TrackRecord] = {}

    def update(self, det: Detection, frame_idx: int) -> TrackRecord:
        rec = self.tracks.get(det.track_id)
        if rec is None:
            rec = TrackRecord(det.track_id, frame_idx, frame_idx)
            self.tracks[det.track_id] = rec
        rec.last_frame = frame_idx
        rec.frames_seen += 1
        rec.class_votes[det.cls] += 1
        x, y = det.bottom_center
        rec.centroids.append((frame_idx, x, y))
        return rec
