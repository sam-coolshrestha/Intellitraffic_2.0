"""Speed accuracy + overspeed classification.

Inputs:
  --records  outputs/vehicle_records.csv   (from the pipeline)
  --gt       gt_speeds.csv                 columns: vehicle_id, gt_speed_kmh
Outputs (research/results/): speed_metrics.csv, speed_pairs.csv, bland_altman.png/.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from research.metrics import bland_altman, confusion, mae, mape, prf, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="outputs/vehicle_records.csv")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--limit", type=float, default=60.0)
    ap.add_argument("--out", default="research/results")
    ap.add_argument("--tag", default="run", help="label for this configuration")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.records).merge(pd.read_csv(a.gt), on="vehicle_id").dropna(subset=["avg_speed_kmh", "gt_speed_kmh"])
    if df.empty:
        raise SystemExit("No overlapping vehicle_ids with valid speeds.")
    est, gt = df["avg_speed_kmh"].to_numpy(), df["gt_speed_kmh"].to_numpy()
    bias, lo, hi = bland_altman(est, gt)
    cm = confusion(est > a.limit, gt > a.limit)
    row = {"tag": a.tag, "n": len(df), "mae_kmh": mae(est, gt), "rmse_kmh": rmse(est, gt), "mape_pct": mape(est, gt),
           "bias_kmh": bias, "loa_low": lo, "loa_high": hi, **cm, **prf(cm["tp"], cm["fp"], cm["fn"])}
    m = pd.DataFrame([row])
    mp = out / "speed_metrics.csv"
    m.to_csv(mp, mode="a", header=not mp.exists(), index=False)
    df[["vehicle_id", "avg_speed_kmh", "gt_speed_kmh"]].to_csv(out / f"speed_pairs_{a.tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    mean = (est + gt) / 2
    ax.scatter(mean, est - gt, s=18)
    for y, ls in [(bias, "-"), (lo, "--"), (hi, "--")]:
        ax.axhline(y, color="k", ls=ls, lw=1)
    ax.set_xlabel("Mean of estimated and true speed (km/h)"); ax.set_ylabel("Estimated - true (km/h)")
    ax.set_title(f"Bland-Altman ({a.tag})"); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"bland_altman_{a.tag}.{ext}", dpi=200)
    print(m.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
