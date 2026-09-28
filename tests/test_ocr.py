from app.ocr import PlateVoter, clean_plate_text, correct_confusions, is_valid_plate


def test_clean():
    assert clean_plate_text("mh 12-ab.1234 ") == "MH12AB1234"


def test_confusions_fixed_positionally():
    assert correct_confusions("MH12ABI234") == "MH12AB1234"   # I -> 1 in the digit block
    assert correct_confusions("M012AB1234") == "MO12AB1234"   # 0 -> O in the state letters


def test_validator():
    assert is_valid_plate("MH12AB1234")
    assert not is_valid_plate("HELLO")


def test_voting_prefers_consistent_valid_plate():
    v = PlateVoter()
    v.add(1, "MH12AB1234", 0.8)
    v.add(1, "MH12AB1234", 0.7)
    v.add(1, "MH12A81234", 0.9)
    text, conf = v.best(1)
    assert text == "MH12AB1234" and conf > 0.7
    assert v.best(99) == ("", 0.0)
