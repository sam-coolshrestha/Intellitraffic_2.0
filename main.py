"""CLI entry point:  python main.py --input traffic.mp4 --output outputs/"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import load_config
from app.pipeline import process_video


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="IntelliTraffic AI - headless traffic video analysis")
    p.add_argument("--input", required=True, help="path to input video")
    p.add_argument("--output", default="outputs", help="output directory")
    p.add_argument("--config", default=None, help="YAML config (default configs/config.yaml)")
    p.add_argument("--conf", type=float, help="detection confidence threshold")
    p.add_argument("--speed-limit", type=float, help="overspeed limit in km/h")
    p.add_argument("--mpp", type=float, help="meters per pixel (simple calibration)")
    p.add_argument("--no-ocr", action="store_true", help="disable plate OCR")
    p.add_argument("--device", help="auto | cpu | cuda:0")
    p.add_argument("--max-frames", type=int, help="process only the first N frames")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args(argv)
    ov: dict = {"model": {}, "speed": {}, "ocr": {}, "video": {}}
    if a.conf is not None: ov["model"]["conf"] = a.conf
    if a.device: ov["model"]["device"] = a.device
    if a.speed_limit is not None: ov["speed"]["limit_kmh"] = a.speed_limit
    if a.mpp is not None: ov["speed"]["meters_per_pixel"] = a.mpp
    if a.no_ocr: ov["ocr"]["enabled"] = False
    if a.max_frames: ov["video"]["max_frames"] = a.max_frames
    cfg = load_config(a.config, ov)

    if not Path(a.input).exists():
        print(f"Input not found: {a.input}", file=sys.stderr)
        return 2
    try:
        def cb(i, n, _f):
            if i % 30 == 0:
                print(f"  frame {i}/{n if n > 0 else '?'}")
        res = process_video(a.input, cfg, a.output, progress_cb=cb)
    except Exception as exc:
        logging.exception("Processing failed: %s", exc)
        return 1

    rec, m = res["records"], res["metadata"]
    print("\n=== IntelliTraffic summary ===")
    print(f"Frames: {m['frames_processed']}  Time: {m['processing_time_s']}s ({m['processing_fps']} fps)")
    print(f"Vehicles: {len(rec)}  Violations: {len(res['violations'])}")
    if len(rec) and rec['avg_speed_kmh'].notna().any():
        print(f"Avg speed: {rec['avg_speed_kmh'].mean():.1f} km/h  Max: {rec['max_speed_kmh'].max():.1f} km/h")
    print(f"Outputs in: {Path(a.output).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
