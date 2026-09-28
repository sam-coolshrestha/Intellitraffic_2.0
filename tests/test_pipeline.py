import json

import pytest

from app.config import load_config
from app.pipeline import process_video
from app.utils import is_h264
from tests.conftest import FakeDetector


def test_pipeline_smoke(synthetic_video, tmp_path):
    # synthetic car moves 5 px/frame @30fps; mpp=0.1 -> 15 m/s = 54 km/h
    cfg = load_config(overrides={
        "speed": {"meters_per_pixel": 0.1, "limit_kmh": 40, "min_frames": 4, "smoothing": 1.0, "consecutive_frames": 3},
        "ocr": {"enabled": False},
    })
    res = process_video(synthetic_video, cfg, tmp_path, detector=FakeDetector())
    for name in ["final_output.mp4", "vehicle_records.csv", "violations.csv", "trajectories.csv",
                 "speed_timeline.csv", "run_metadata.json", "trajectory_heatmap.png"]:
        assert (tmp_path / name).exists(), name
    assert is_h264(tmp_path / "final_output.mp4")
    rec = res["records"]
    assert len(rec) == 1 and rec.iloc[0]["class"] == "car"
    assert rec.iloc[0]["avg_speed_kmh"] == pytest.approx(54.0, rel=0.05)
    assert bool(rec.iloc[0]["overspeed"]) and len(res["violations"]) == 1
    assert (tmp_path / "snapshots" / "vehicle_1.jpg").exists()
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert meta["frames_processed"] == 60 and "stage_times_s" in meta


def test_bad_video_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        process_video(bad, load_config(), tmp_path / "o", detector=FakeDetector())


def test_config_overrides():
    cfg = load_config(overrides={"speed": {"limit_kmh": 99}})
    assert cfg.speed.limit_kmh == 99 and cfg.model.imgsz == 640
