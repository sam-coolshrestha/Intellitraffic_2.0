from app.violations import ViolationDetector


def test_needs_consecutive_frames():
    v = ViolationDetector(60, consecutive=3)
    assert not v.update(1, 70, 0)
    assert not v.update(1, 70, 1)
    assert v.update(1, 70, 2)          # third consecutive
    assert not v.update(1, 80, 3)      # only flagged once
    assert 1 in v.flagged


def test_streak_resets():
    v = ViolationDetector(60, consecutive=3)
    v.update(1, 70, 0); v.update(1, 70, 1); v.update(1, 50, 2)
    assert not v.update(1, 70, 3)
    assert 1 not in v.flagged


def test_none_speed_ignored():
    v = ViolationDetector(60, consecutive=1)
    assert not v.update(1, None, 0)
