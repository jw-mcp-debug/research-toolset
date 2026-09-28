"""Fixed codes for LLM-returned ratings."""

import pytest

from src.core.vocab import normalize_level, normalize_quality
from src.pipeline.harvest_parser import parse_harvest_response_with_polarity


@pytest.mark.parametrize("raw,code", [
    ("hoch", "high"), ("High", "high"), (" mittel ", "medium"), ("medium", "medium"),
    ("niedrig", "low"), ("LOW", "low"), ('"hoch"', "high"),
])
def test_levels_accept_german_and_english(raw, code):
    assert normalize_level(raw) == code


def test_unknown_level_falls_back_to_default():
    assert normalize_level("very_high") == "medium"
    assert normalize_level("very_high", default=None) is None


@pytest.mark.parametrize("raw,code", [
    ("stark", "strong"), ("Moderate (small sample)", "moderate"),
    ("schwach", "weak"), ("", "unknown"), ("n/a", "unknown"),
])
def test_quality(raw, code):
    assert normalize_quality(raw) == code


@pytest.mark.parametrize("marker,polarity", [
    ("POSITIV", "positive"), ("POSITIVE", "positive"),
    ("NEGATIV", "negative"), ("NEGATIVE", "negative"), ("META", "meta"),
])
def test_harvest_markers_in_both_languages(marker, polarity):
    resp = f"[F1] [{marker}] Fact: something\nReliability: high\n"
    ex = parse_harvest_response_with_polarity(resp, "u", "t")
    assert ex and ex[0].polarity == polarity and ex[0].reliability == "high"
