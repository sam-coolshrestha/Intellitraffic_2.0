"""Detection precision / recall / mAP with Ultralytics validation on a YOLO-format dataset.

  python -m research.evaluate_detection --data data/vehicles.yaml --weights yolov8n.pt yolov8s.pt --imgsz 640
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset yaml (YOLO format)")
    ap.add_argument("--weights", nargs="+", default=["yolov8n.pt"])
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="research/results")
    a = ap.parse_args()
    from ultralytics import YOLO

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in a.weights:
        model = YOLO(w)
        for sz in a.imgsz:
            r = model.val(data=a.data, imgsz=sz, device=a.device, verbose=False, plots=False)
            rows.append({"weights": w, "imgsz": sz, "precision": r.box.mp, "recall": r.box.mr,
                         "map50": r.box.map50, "map50_95": r.box.map})
    df = pd.DataFrame(rows)
    df.to_csv(out / "detection_metrics.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
