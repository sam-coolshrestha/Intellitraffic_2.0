"""Ablation grid over model size x image size x calibration x OCR.

  python -m research.ablation --video clip.mp4 --gt gt_speeds.csv --mpp 0.05 \
      --weights yolov8n.pt yolov8s.pt --imgsz 480 640 --ocr on off

Writes research/results/ablation.csv (+ ablation.png). --gt (vehicle_id, gt_speed_kmh) is optional.
"""
import argparse
import itertools
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.config import load_config
from app.pipeline import process_video
from research.metrics import mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--mpp", type=float, default=0.05)
    ap.add_argument("--weights", nargs="+", default=["yolov8n.pt"])
    ap.add_argument("--imgsz", type=int, nargs="+", default=[480, 640, 960])
    ap.add_argument("--ocr", nargs="+", default=["off"], choices=["on", "off"])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default="research/results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(a.gt) if a.gt else None

    rows = []
    for w, sz, ocr in itertools.product(a.weights, a.imgsz, a.ocr):
        cfg = load_config(overrides={"model": {"weights": f"models/{w}", "imgsz": sz},
                                     "speed": {"meters_per_pixel": a.mpp},
                                     "ocr": {"enabled": ocr == "on"}, "video": {"max_frames": a.max_frames}})
        with tempfile.TemporaryDirectory() as tmp:
            res = process_video(a.video, cfg, tmp)
        rec, meta = res["records"], res["metadata"]
        row = {"weights": w, "imgsz": sz, "ocr": ocr, "vehicles": len(rec), "violations": len(res["violations"]),
               "plates_read": int((rec["plate"].fillna("") != "").sum()), "fps": meta["processing_fps"]}
        if gt is not None:
            j = rec.merge(gt, on="vehicle_id").dropna(subset=["avg_speed_kmh"])
            row["speed_mae_kmh"] = mae(j["avg_speed_kmh"], j["gt_speed_kmh"]) if len(j) else float("nan")
        rows.append(row)
        print(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    for w, g in df.groupby("weights"):
        ax.plot(g["imgsz"], g["fps"], marker="o", label=w)
    ax.set_xlabel("Image size"); ax.set_ylabel("Processing FPS"); ax.legend(); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ablation_fps.{ext}", dpi=200)


if __name__ == "__main__":
    main()
