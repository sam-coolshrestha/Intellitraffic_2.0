import pandas as pd

from app import analytics as an


def frame():
    return pd.DataFrame({
        "vehicle_id": [1, 2, 3], "class": ["car", "bus", "car"], "plate": ["MH12AB1234", "", "DL1CAB9999"],
        "first_frame": [0, 0, 5], "last_frame": [50, 60, 70], "duration_s": [2, 2, 2],
        "avg_speed_kmh": [40.0, 30.0, 70.0], "max_speed_kmh": [45.0, 35.0, 80.0],
        "overspeed": [False, False, True], "snapshot_path": ["", "", "x.jpg"],
    })


def test_kpis():
    k = an.kpis(frame())
    assert k["total"] == 3 and k["violations"] == 1 and k["max_speed"] == 80.0


def test_plate_search_partial_case_insensitive():
    assert list(an.filter_records(frame(), plate_query="ab12").vehicle_id) == [1]
    assert list(an.filter_records(frame(), plate_query="dl1c-ab").vehicle_id) == [3]


def test_filters_combine():
    out = an.filter_records(frame(), classes=["car"], speed_range=(50, 100), only_violations=True)
    assert list(out.vehicle_id) == [3]


def test_empty_csv(tmp_path):
    p = tmp_path / "r.csv"
    assert an.load_records(p).empty                 # missing
    p.write_text("")
    assert an.load_records(p).empty                 # zero bytes
    assert an.kpis(an.load_records(p))["total"] == 0
    assert an.type_distribution(an.empty_records()).empty
    assert "IntelliTraffic" in an.summary_html(an.empty_records(), 60)


def test_roundtrip(tmp_path):
    p = tmp_path / "r.csv"
    frame().to_csv(p, index=False)
    assert len(an.load_records(p)) == 3
