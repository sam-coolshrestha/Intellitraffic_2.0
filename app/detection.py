"""YOLOv8 detection + tracking wrapper with strict vehicle-class filtering."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from .config import ROOT, ModelCfg

log = logging.getLogger("intellitraffic")

# COCO class ids for vehicles. Everything else (umbrella, person, ...) is dropped.
COCO_VEHICLES = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}


@dataclass
class Detection:
    track_id: int
    cls: str
    conf: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def bottom_center(self) -> Tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, float(y2)


def filter_detections(dets: Iterable[Detection], allowed: Iterable[str]) -> List[Detection]:
    """Keep only detections whose class name is in `allowed` (defence in depth)."""
    allowed_set = {a.lower() for a in allowed}
    return [d for d in dets if d.cls.lower() in allowed_set]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class VehicleDetector:
    """Detect and track vehicles. Loads the model lazily (torch is imported only when needed)."""

    def __init__(self, cfg: ModelCfg):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.class_ids = [COCO_VEHICLES[c] for c in cfg.classes if c in COCO_VEHICLES]
        if not self.class_ids:
            raise ValueError(f"No valid vehicle classes in {cfg.classes}; choose from {list(COCO_VEHICLES)}")
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            weights = Path(self.cfg.weights)
            if not weights.is_absolute():
                weights = ROOT / weights
            # If the file is missing, pass the bare name so ultralytics downloads it.
            self._model = YOLO(str(weights) if weights.exists() else weights.name)
            log.info("Loaded YOLO model %s on %s", weights.name, self.device)
        return self._model

    def reset(self) -> None:
        """Reset tracker state (call between videos)."""
        model = self._model
        if model is not None and getattr(model, "predictor", None) is not None:
            model.predictor = None

    def track(self, frame: np.ndarray) -> List[Detection]:
        model = self._load()
        results = model.track(
            frame, persist=True, tracker=self.cfg.tracker, classes=self.class_ids,
            conf=self.cfg.conf, iou=self.cfg.iou, imgsz=self.cfg.imgsz,
            device=self.device, verbose=False,
        )
        r = results[0]
        boxes = r.boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int)
        names = r.names
        dets = [
            Detection(int(i), str(names[int(c)]), float(p), tuple(int(v) for v in b))
            for b, p, c, i in zip(xyxy, conf, cls, ids)
        ]
        return filter_detections(dets, self.cfg.classes)
