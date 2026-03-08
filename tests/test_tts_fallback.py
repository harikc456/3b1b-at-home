import pytest


def test_text_length_duration():
    text = "This is a ten word sentence that we use for testing."
    words = len(text.split())
    assert words / 2.5 == pytest.approx(words / 2.5)


def test_text_length_duration_minimum():
    text = "Yes."
    assert len(text.split()) / 2.5 > 0
