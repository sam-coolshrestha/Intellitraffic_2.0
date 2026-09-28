"""Latency and throughput per pipeline stage, CPU vs GPU, plus peak memory.

  python -m research.benchmark --video clip.mp4 --devices cpu cuda:0 --max-frames 300
"""
import argparse
import platform
import resource
import tempfile
from pathlib import Path

import pandas as pd

from app.config import load_config
from app.pipeline import process_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--devices", nargs="+", default=["cpu"])
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", default="research/results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for dev in a.devices:
        cfg = load_config(overrides={"model": {"weights": f"models/{a.weights}", "imgsz": a.imgsz, "device": dev},
                                     "ocr": {"enabled": a.ocr}, "video": {"max_frames": a.max_frames}})
        with tempfile.TemporaryDirectory() as tmp:
            meta = process_video(a.video, cfg, tmp)["metadata"]
        n = meta["frames_processed"]
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024
        row = {"device": dev, "frames": n, "fps": meta["processing_fps"], "peak_ram_mb": round(peak_mb, 1)}
        row.update({f"ms_per_frame_{k}": round(v / n * 1000, 2) for k, v in meta["stage_times_s"].items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "benchmark.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
