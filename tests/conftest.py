import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.detection import Detection  # noqa: E402

W, H, FPS, N, STEP = 320, 240, 30, 60, 10  # rectangle moves 10 px/frame


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("vid") / "synthetic.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        f = np.full((H, W, 3), 60, np.uint8)
        x = 10 + i * STEP // 2  # 5 px/frame in the video itself (kept inside frame)
        cv2.rectangle(f, (x, 100), (x + 40, 140), (200, 200, 200), -1)
        w.write(f)
    w.release()
    return path


class FakeDetector:
    """Follows the synthetic rectangle: 5 px/frame, plus a stray non-vehicle detection."""

    def __init__(self):
        self.i = 0

    def track(self, frame):
        x = 10 + self.i * STEP // 2
        self.i += 1
        return [Detection(1, "car", 0.9, (x, 100, x + 40, 140))]
