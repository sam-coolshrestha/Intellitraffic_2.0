"""Metric helpers shared by the research scripts."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def mae(est, gt) -> float:
    return float(np.mean(np.abs(np.asarray(est, float) - np.asarray(gt, float))))


def rmse(est, gt) -> float:
    return float(np.sqrt(np.mean((np.asarray(est, float) - np.asarray(gt, float)) ** 2)))


def mape(est, gt) -> float:
    gt = np.asarray(gt, float)
    est = np.asarray(est, float)
    m = gt != 0
    return float(np.mean(np.abs(est[m] - gt[m]) / np.abs(gt[m])) * 100)


def bland_altman(est, gt) -> Tuple[float, float, float]:
    """Return (mean difference, lower limit, upper limit of agreement)."""
    d = np.asarray(est, float) - np.asarray(gt, float)
    m, s = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0
    return m, m - 1.96 * s, m + 1.96 * s


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, truth: str) -> float:
    """Character error rate."""
    return levenshtein(pred, truth) / max(1, len(truth))


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def confusion(pred: Sequence[bool], truth: Sequence[bool]) -> Dict[str, int]:
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    return {"tp": int((pred & truth).sum()), "fp": int((pred & ~truth).sum()),
            "fn": int((~pred & truth).sum()), "tn": int((~pred & ~truth).sum())}
