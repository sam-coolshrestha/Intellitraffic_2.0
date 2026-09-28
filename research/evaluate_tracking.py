"""Tracking metrics (MOTA, MOTP, IDF1, ID switches) with motmetrics; compares ByteTrack vs BoT-SORT.

  python -m research.evaluate_tracking --video clip.mp4 --gt gt.txt --trackers bytetrack.yaml botsort.yaml

--gt is MOT-format ground truth:  frame,id,x,y,w,h,conf,-1,-1,-1   (frames 1-indexed)
"""
import argparse
from pathlib import Path

import cv2
import motmetrics as mm
import pandas as pd

from app.detection import COCO_VEHICLES


def run_tracker(video: str, weights: str, tracker: str, conf: float, imgsz: int):
    from ultralytics import YOLO

    model = YOLO(weights)
    rows, cap, f = [], cv2.VideoCapture(video), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f += 1
        r = model.track(frame, persist=True, tracker=tracker, classes=list(COCO_VEHICLES.values()),
                        conf=conf, imgsz=imgsz, verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            for (x1, y1, x2, y2), i in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy()):
                rows.append((f, int(i), x1, y1, x2 - x1, y2 - y1))
    cap.release()
    return pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h"])


def score(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    acc = mm.MOTAccumulator(auto_id=False)
    for f in sorted(set(gt["frame"]) | set(pred["frame"])):
        g, p = gt[gt["frame"] == f], pred[pred["frame"] == f]
        d = mm.distances.iou_matrix(g[["x", "y", "w", "h"]].to_numpy(), p[["x", "y", "w", "h"]].to_numpy(), max_iou=0.5)
        acc.update(g["id"].tolist(), p["id"].tolist(), d, frameid=int(f))
    s = mm.metrics.create().compute(acc, metrics=["mota", "motp", "idf1", "num_switches", "num_false_positives",
                                                  "num_misses"], name="x")
    return {k: float(v) for k, v in s.iloc[0].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--trackers", nargs="+", default=["bytetrack.yaml", "botsort.yaml"])
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="research/results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(a.gt, header=None).iloc[:, :6]
    gt.columns = ["frame", "id", "x", "y", "w", "h"]
    rows = []
    for t in a.trackers:
        pred = run_tracker(a.video, a.weights, t, a.conf, a.imgsz)
        rows.append({"tracker": t, **score(gt, pred)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "tracking_metrics.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
