"""License plate localisation, OCR (EasyOCR), text cleanup and multi-frame voting."""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import OcrCfg

log = logging.getLogger("intellitraffic")

# Indian format, e.g. MH12AB1234 / DL1CAB1234 (soft validator, not a hard filter)
PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def clean_plate_text(text: str) -> str:
    """Uppercase and strip everything except A-Z / 0-9."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_plate(text: str) -> bool:
    return bool(PLATE_RE.match(text))


def correct_confusions(text: str) -> str:
    """Positional O/0, I/1, S/5 fixes for the Indian plate layout (state letters, RTO digits, series, 4 digits)."""
    t = list(clean_plate_text(text))
    if len(t) < 8 or len(t) > 11:
        return "".join(t)
    for i in range(len(t) - 4, len(t)):        # last four are digits
        t[i] = _TO_DIGIT.get(t[i], t[i])
    for i in range(2):                          # first two are state letters
        t[i] = _TO_LETTER.get(t[i], t[i])
    t[2] = _TO_DIGIT.get(t[2], t[2])            # RTO code starts with a digit
    return "".join(t)


class PlateVoter:
    """Confidence-weighted majority vote of cleaned plate strings per vehicle id."""

    def __init__(self) -> None:
        self._w: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._c: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    def add(self, track_id: int, text: str, conf: float) -> None:
        text = correct_confusions(text)
        if len(text) < 4:
            return
        weight = conf * (1.5 if is_valid_plate(text) else 1.0)
        self._w[track_id][text] += weight
        self._c[track_id][text].append(conf)

    def best(self, track_id: int) -> Tuple[str, float]:
        """Return (plate, mean_confidence); ('', 0.0) if nothing was read."""
        cand = self._w.get(track_id)
        if not cand:
            return "", 0.0
        text = max(cand, key=cand.get)
        confs = self._c[track_id][text]
        return text, float(sum(confs) / len(confs))


def find_plate_region(vehicle_crop: np.ndarray) -> np.ndarray:
    """Classical plate localisation (edges + contour aspect ratio). Falls back to the lower part of the crop."""
    h, w = vehicle_crop.shape[:2]
    if h < 20 or w < 20:
        return vehicle_crop
    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0:
            continue
        aspect, area = cw / ch, cw * ch
        if 2.0 <= aspect <= 6.0 and 0.02 * h * w <= area <= 0.4 * h * w and y > 0.3 * h:
            if area > best_area:
                best, best_area = (x, y, cw, ch), area
    if best is not None:
        x, y, cw, ch = best
        return vehicle_crop[y:y + ch, x:x + cw]
    return vehicle_crop[int(h * 0.55):, :]


def preprocess_plate(img: np.ndarray) -> np.ndarray:
    """Upscale + grayscale + CLAHE for OCR."""
    if img.size == 0:
        return img
    h, w = img.shape[:2]
    scale = max(1.0, 200.0 / max(w, 1))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


class PlateReader:
    """EasyOCR wrapper. The reader is created once, lazily."""

    def __init__(self, cfg: OcrCfg, gpu: bool = False):
        self.cfg = cfg
        self.gpu = gpu
        self._reader = None

    def _get(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self.cfg.languages, gpu=self.gpu, verbose=False)
        return self._reader

    def read(self, vehicle_crop: np.ndarray) -> Optional[Tuple[str, float]]:
        """Return (raw_text, confidence) or None if nothing usable was read."""
        if min(vehicle_crop.shape[:2]) < self.cfg.min_crop_px:
            return None
        region = preprocess_plate(find_plate_region(vehicle_crop))
        if region.size == 0:
            return None
        try:
            results = self._get().readtext(region, allowlist=ALLOWLIST, detail=1, paragraph=False)
        except Exception as exc:  # OCR must never crash the pipeline
            log.warning("OCR failed: %s", exc)
            return None
        parts = [(clean_plate_text(t), float(c)) for _, t, c in results if clean_plate_text(t)]
        if not parts:
            return None
        text = "".join(p[0] for p in parts)
        conf = sum(p[1] for p in parts) / len(parts)
        return (text, conf) if conf >= self.cfg.min_confidence else None
