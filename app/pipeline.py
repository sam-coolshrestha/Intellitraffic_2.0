"""End-to-end video processing: detect -> track -> speed -> OCR -> violations -> outputs."""
from __future__ import annotations

import json
import logging
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from . import __version__
from .analytics import RECORD_COLUMNS
from .config import ROOT, Config
from .detection import VehicleDetector, resolve_device
from .ocr import PlateReader, PlateVoter
from .speed import SpeedEstimator
from .tracking import TrackHistory
from .trajectory import TrajectoryStore
from .utils import color_for_id, make_writer, reencode_h264, video_info
from .violations import ViolationDetector

log = logging.getLogger("intellitraffic")
ProgressCb = Callable[[int, int, np.ndarray], None]


def _git_hash() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip() or None
    except Exception:
        return None


def _label(text: str, frame: np.ndarray, x: int, y: int, color) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y0 = max(th + 4, y)
    cv2.rectangle(frame, (x, y0 - th - 4), (x + tw + 4, y0 + base - 2), color, -1)
    cv2.putText(frame, text, (x + 2, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def process_video(
    input_path: str | Path,
    cfg: Config,
    output_dir: str | Path,
    detector: Optional[Any] = None,
    ocr_reader: Optional[Any] = None,
    progress_cb: Optional[ProgressCb] = None,
) -> Dict[str, Any]:
    """Run the full pipeline. `detector` needs a `.track(frame) -> list[Detection]` method
    (lets tests inject a fake). Returns paths and dataframes."""
    t_start = time.perf_counter()
    out = Path(output_dir)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)

    fps, width, height, total = video_info(input_path)
    fps = float(cfg.video.fps_override or fps or 25.0)
    if cfg.video.max_frames:
        total = min(total, int(cfg.video.max_frames)) if total > 0 else int(cfg.video.max_frames)

    detector = detector or VehicleDetector(cfg.model)
    if hasattr(detector, "reset"):
        detector.reset()
    reader = None
    if cfg.ocr.enabled:
        reader = ocr_reader or PlateReader(cfg.ocr, gpu=resolve_device(cfg.model.device).startswith("cuda"))

    speed = SpeedEstimator(cfg.speed, fps)
    viol = ViolationDetector(cfg.speed.limit_kmh, cfg.speed.consecutive_frames)
    traj, hist, voter = TrajectoryStore(), TrackHistory(), PlateVoter()
    speeds: Dict[int, List[float]] = defaultdict(list)
    timeline: List[Dict[str, Any]] = []
    snapshots: Dict[int, str] = {}
    stage = defaultdict(float)

    raw_path = out / "_raw.mp4"
    final_path = out / "final_output.mp4"
    writer = make_writer(raw_path, fps, (width, height))
    cap = cv2.VideoCapture(str(input_path))
    first_frame = None
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (cfg.video.max_frames and idx >= cfg.video.max_frames):
                break
            if first_frame is None:
                first_frame = frame.copy()

            t0 = time.perf_counter()
            dets = detector.track(frame)
            stage["detect_track"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            current: List[Any] = []
            for d in dets:
                x, y = d.bottom_center
                hist.update(d, idx)
                traj.add(d.track_id, idx, x, y)
                v = speed.update(d.track_id, idx, (x, y))
                if v is not None:
                    speeds[d.track_id].append(v)
                    timeline.append({"frame": idx, "time_s": round(idx / fps, 3),
                                     "vehicle_id": d.track_id, "speed_kmh": round(v, 2)})
                current.append((d, v))
            stage["speed"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if reader is not None:
                for d, _ in current:
                    seen = hist.tracks[d.track_id].frames_seen
                    if seen == 1 or idx % max(1, cfg.ocr.every_n_frames) == 0:
                        x1, y1, x2, y2 = d.bbox
                        crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                        if crop.size:
                            res = reader.read(crop)
                            if res:
                                voter.add(d.track_id, res[0], res[1])
            stage["ocr"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            for d, v in current:
                if viol.update(d.track_id, v, idx):
                    x1, y1, x2, y2 = d.bbox
                    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                    if crop.size:
                        p = out / "snapshots" / f"vehicle_{d.track_id}.jpg"
                        cv2.imwrite(str(p), crop)
                        snapshots[d.track_id] = str(p)

            traj.draw(frame, [d.track_id for d, _ in current], cfg.video.trail_length)
            for d, v in current:
                bad = d.track_id in viol.flagged
                color = (0, 0, 255) if bad else color_for_id(d.track_id)
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if bad else 2)
                plate, _ = voter.best(d.track_id)
                parts = [f"#{d.track_id} {hist.tracks[d.track_id].cls}"]
                if v is not None:
                    parts.append(f"{v:.0f}km/h")
                if plate:
                    parts.append(plate)
                _label(" ".join(parts), frame, x1, y1, color)
            cv2.putText(frame, f"Limit {cfg.speed.limit_kmh:.0f} km/h | Violations {len(viol.flagged)}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            stage["annotate_write"] += time.perf_counter() - t0

            if progress_cb and idx % max(1, cfg.video.progress_every) == 0:
                progress_cb(idx, total, frame)
            idx += 1
    finally:
        cap.release()
        writer.release()

    if idx == 0:
        raise ValueError("No frames could be read from the video.")

    if reencode_h264(raw_path, final_path):
        raw_path.unlink(missing_ok=True)
    else:  # keep a playable-in-most-players fallback
        raw_path.replace(final_path)

    # ---- records ----
    rows = []
    for tid, rec in hist.tracks.items():
        if rec.frames_seen < cfg.speed.min_frames:
            continue  # drop spurious short tracks
        sp = speeds.get(tid, [])
        plate, _ = voter.best(tid)
        rows.append({
            "vehicle_id": tid, "class": rec.cls, "plate": plate,
            "first_frame": rec.first_frame, "last_frame": rec.last_frame,
            "duration_s": round((rec.last_frame - rec.first_frame + 1) / fps, 2),
            "avg_speed_kmh": round(float(np.mean(sp)), 2) if sp else np.nan,
            "max_speed_kmh": round(float(np.max(sp)), 2) if sp else np.nan,
            "overspeed": tid in viol.flagged,
            "snapshot_path": snapshots.get(tid, ""),
        })
    records = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    violations = records[records["overspeed"]].copy()
    trajectories = traj.to_dataframe()
    speed_df = pd.DataFrame(timeline, columns=["frame", "time_s", "vehicle_id", "speed_kmh"])

    records.to_csv(out / "vehicle_records.csv", index=False)
    violations.to_csv(out / "violations.csv", index=False)
    trajectories.to_csv(out / "trajectories.csv", index=False)
    speed_df.to_csv(out / "speed_timeline.csv", index=False)
    heat_path = out / "trajectory_heatmap.png"
    if first_frame is not None:
        cv2.imwrite(str(heat_path), traj.heatmap(first_frame))

    elapsed = time.perf_counter() - t_start
    meta = {
        "version": __version__, "git_commit": _git_hash(), "input": str(input_path),
        "fps": fps, "resolution": [width, height], "frames_processed": idx,
        "processing_time_s": round(elapsed, 2), "processing_fps": round(idx / elapsed, 2),
        "stage_times_s": {k: round(v, 3) for k, v in stage.items()},
        "device": resolve_device(cfg.model.device), "config": cfg.to_dict(),
    }
    try:
        import ultralytics
        meta["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    return {
        "video": final_path, "records": records, "violations": violations,
        "trajectories": trajectories, "speed_timeline": speed_df,
        "heatmap": heat_path if heat_path.exists() else None,
        "metadata": meta, "output_dir": out,
    }
