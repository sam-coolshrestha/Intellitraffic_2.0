"""Central configuration loaded from YAML into typed dataclasses."""
from __future__ import annotations

import copy
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


@dataclass
class ModelCfg:
    weights: str = "models/yolov8n.pt"
    imgsz: int = 640
    conf: float = 0.35
    iou: float = 0.5
    device: str = "auto"
    classes: List[str] = field(default_factory=lambda: ["car", "motorcycle", "bus", "truck"])
    tracker: str = "bytetrack.yaml"


@dataclass
class SpeedCfg:
    mode: str = "simple"
    meters_per_pixel: float = 0.05
    image_points: List[List[float]] = field(default_factory=list)
    world_points: List[List[float]] = field(default_factory=list)
    window: int = 8
    min_frames: int = 6
    smoothing: float = 0.4
    limit_kmh: float = 60.0
    consecutive_frames: int = 5


@dataclass
class OcrCfg:
    enabled: bool = True
    languages: List[str] = field(default_factory=lambda: ["en"])
    every_n_frames: int = 10
    min_crop_px: int = 60
    min_confidence: float = 0.30


@dataclass
class VideoCfg:
    fps_override: Optional[float] = None
    max_frames: Optional[int] = None
    trail_length: int = 30
    progress_every: int = 5


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    speed: SpeedCfg = field(default_factory=SpeedCfg)
    ocr: OcrCfg = field(default_factory=OcrCfg)
    video: VideoCfg = field(default_factory=VideoCfg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build(cls, data: Dict[str, Any]):
    """Build a dataclass from a dict, ignoring unknown keys and recursing into sections."""
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.default_factory is not MISSING:
            default = f.default_factory()
            if is_dataclass(default) and isinstance(value, dict):
                value = _build(type(default), value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load config from YAML (default configs/config.yaml) with optional nested overrides."""
    data: Dict[str, Any] = {}
    p = Path(path) if path else DEFAULT_CONFIG
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}
    if overrides:
        data = _deep_update(data, overrides)
    return _build(Config, data)
