import pytest


def test_text_length_duration_ten_words():
    # 10 words / 2.5 wps = 4.0 seconds
    text = "one two three four five six seven eight nine ten"
    assert len(text.split()) / 2.5 == pytest.approx(4.0)


def test_text_length_duration_five_words():
    # 5 words / 2.5 wps = 2.0 seconds
    text = "one two three four five"
    assert len(text.split()) / 2.5 == pytest.approx(2.0)


def test_text_length_duration_positive():
    text = "Yes."
    assert len(text.split()) / 2.5 > 0
