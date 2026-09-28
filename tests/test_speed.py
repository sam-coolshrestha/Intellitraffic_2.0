
import pytest

from app.config import SpeedCfg
from app.speed import SpeedEstimator


def run(est, px_per_frame, frames=20, y=100):
    v = None
    for f in range(frames):
        v = est.update(1, f, (f * px_per_frame, y))
    return v


def test_simple_speed_matches_expected():
    # 10 px/frame * 30 fps * 0.05 m/px = 15 m/s = 54 km/h
    est = SpeedEstimator(SpeedCfg(mode="simple", meters_per_pixel=0.05, min_frames=4, smoothing=1.0), fps=30)
    assert run(est, 10) == pytest.approx(54.0, rel=1e-3)


def test_warmup_returns_none():
    est = SpeedEstimator(SpeedCfg(min_frames=6), fps=30)
    assert est.update(1, 0, (0, 0)) is None


def test_homography_speed():
    # image square 100x100 px maps to 10x10 m -> 0.1 m/px -> 10 px/frame @30fps = 30 m/s = 108 km/h
    cfg = SpeedCfg(mode="homography", min_frames=4, smoothing=1.0,
                   image_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
                   world_points=[[0, 0], [10, 0], [10, 10], [0, 10]])
    est = SpeedEstimator(cfg, fps=30)
    v = None
    for f in range(8):
        v = est.update(1, f, (f * 10, 50))
    assert v == pytest.approx(108.0, rel=1e-3)


def test_bad_homography_points_raise():
    with pytest.raises(ValueError):
        SpeedEstimator(SpeedCfg(mode="homography", image_points=[[0, 0]], world_points=[[0, 0]]), fps=30)


def test_bad_fps_raises():
    with pytest.raises(ValueError):
        SpeedEstimator(SpeedCfg(), fps=0)
