"""Pandas aggregations and report generation for the dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

RECORD_COLUMNS = [
    "vehicle_id", "class", "plate", "first_frame", "last_frame", "duration_s",
    "avg_speed_kmh", "max_speed_kmh", "overspeed", "snapshot_path",
]


def empty_records() -> pd.DataFrame:
    return pd.DataFrame(columns=RECORD_COLUMNS)


def load_records(path: str | Path) -> pd.DataFrame:
    """Load vehicle_records.csv; returns an empty, correctly-typed frame if missing/empty."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return empty_records()
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return empty_records()
    for col in RECORD_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["plate"] = df["plate"].fillna("").astype(str)
    df["overspeed"] = df["overspeed"].fillna(False).astype(bool)
    return df[RECORD_COLUMNS]


def kpis(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"total": 0, "avg_speed": 0.0, "max_speed": 0.0, "violations": 0}
    return {
        "total": int(len(df)),
        "avg_speed": float(df["avg_speed_kmh"].mean(skipna=True) or 0.0),
        "max_speed": float(df["max_speed_kmh"].max(skipna=True) or 0.0),
        "violations": int(df["overspeed"].sum()),
    }


def type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["class", "count"])
    out = df["class"].value_counts().rename_axis("class").reset_index(name="count")
    return out


def filter_records(
    df: pd.DataFrame,
    plate_query: str = "",
    classes: Optional[Iterable[str]] = None,
    speed_range: Optional[Tuple[float, float]] = None,
    only_violations: bool = False,
) -> pd.DataFrame:
    """Partial, case-insensitive plate search plus type / speed / violation filters."""
    out = df
    if plate_query:
        q = "".join(ch for ch in plate_query.upper() if ch.isalnum())
        out = out[out["plate"].str.upper().str.contains(q, regex=False, na=False)]
    if classes:
        out = out[out["class"].isin(list(classes))]
    if speed_range is not None:
        lo, hi = speed_range
        out = out[out["max_speed_kmh"].between(lo, hi)]
    if only_violations:
        out = out[out["overspeed"]]
    return out


def summary_html(df: pd.DataFrame, limit_kmh: float) -> str:
    """One-page HTML summary report (printable to PDF from the browser)."""
    k = kpis(df)
    table = df.drop(columns=["snapshot_path"]).round(1).to_html(index=False, border=0) if not df.empty else "<p>No vehicles.</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>IntelliTraffic report</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;color:#222}}table{{border-collapse:collapse;font-size:13px}}
td,th{{border:1px solid #ddd;padding:4px 8px}}th{{background:#f3f3f3}}.k{{display:inline-block;margin:0 24px 12px 0}}
.k b{{display:block;font-size:24px}}</style></head><body>
<h1>IntelliTraffic AI - Analysis Report</h1>
<div class="k"><b>{k['total']}</b>Vehicles</div><div class="k"><b>{k['avg_speed']:.1f} km/h</b>Average speed</div>
<div class="k"><b>{k['max_speed']:.1f} km/h</b>Max speed</div><div class="k"><b>{k['violations']}</b>Violations (&gt; {limit_kmh:.0f} km/h)</div>
<h2>Vehicles</h2>{table}</body></html>"""
