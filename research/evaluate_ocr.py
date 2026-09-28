"""Plate OCR accuracy: exact match and character error rate.

  --records outputs/vehicle_records.csv
  --gt gt_plates.csv   columns: vehicle_id, gt_plate[, condition]   (condition e.g. day/night)
"""
import argparse
from pathlib import Path

import pandas as pd

from app.ocr import clean_plate_text
from research.metrics import cer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="outputs/vehicle_records.csv")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="research/results")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.records).merge(pd.read_csv(a.gt), on="vehicle_id")
    df["plate"] = df["plate"].fillna("").astype(str).map(clean_plate_text)
    df["gt_plate"] = df["gt_plate"].astype(str).map(clean_plate_text)
    df["exact"] = df["plate"] == df["gt_plate"]
    df["cer"] = [cer(p, g) for p, g in zip(df["plate"], df["gt_plate"])]
    if "condition" not in df:
        df["condition"] = "all"
    res = df.groupby("condition").agg(n=("exact", "size"), exact_match=("exact", "mean"), mean_cer=("cer", "mean")).reset_index()
    res.insert(0, "tag", a.tag)
    rp = out / "ocr_metrics.csv"
    res.to_csv(rp, mode="a", header=not rp.exists(), index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
