import pytest

from research.metrics import bland_altman, cer, confusion, levenshtein, mae, mape, prf, rmse


def test_error_metrics():
    assert mae([50, 60], [40, 60]) == 5
    assert rmse([50, 60], [40, 60]) == pytest.approx(7.0711, rel=1e-3)
    assert mape([50], [40]) == pytest.approx(25.0)


def test_bland_altman():
    m, lo, hi = bland_altman([10, 12, 14], [10, 10, 10])
    assert lo < m < hi and m == pytest.approx(2.0)


def test_text_metrics():
    assert levenshtein("kitten", "sitting") == 3
    assert cer("MH12AB1234", "MH12AB1234") == 0
    assert cer("MH12AB123", "MH12AB1234") == pytest.approx(0.1)


def test_prf_and_confusion():
    cm = confusion([True, True, False, False], [True, False, True, False])
    assert cm == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert prf(1, 1, 1)["f1"] == pytest.approx(0.5)
