import pytest

from app.config import ModelCfg
from app.detection import Detection, VehicleDetector, filter_detections


def test_filter_drops_non_vehicles():
    dets = [Detection(1, "car", .9, (0, 0, 1, 1)), Detection(2, "umbrella", .9, (0, 0, 1, 1)),
            Detection(3, "person", .9, (0, 0, 1, 1)), Detection(4, "truck", .9, (0, 0, 1, 1))]
    out = filter_detections(dets, ["car", "motorcycle", "bus", "truck"])
    assert [d.cls for d in out] == ["car", "truck"]


def test_class_ids_from_config():
    d = VehicleDetector(ModelCfg(classes=["car", "bus"]))
    assert d.class_ids == [2, 5]


def test_no_valid_classes_raises():
    with pytest.raises(ValueError):
        VehicleDetector(ModelCfg(classes=["umbrella"]))
