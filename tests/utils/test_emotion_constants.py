"""Tests for the shared emotion-label constants and normalization helpers."""

from src.utils.emotion_constants import (
    DEFAULT_EMOTION_LABELS,
    DEFAULT_SENTIMENT_SCORES,
    find_label_index_by_name,
    normalize_emotion_labels,
    normalize_sentiment_scores,
)


def test_default_labels_have_seven_classes():
    assert len(DEFAULT_EMOTION_LABELS) == 7
    assert DEFAULT_EMOTION_LABELS[6] == "Other"


def test_every_label_has_a_sentiment_score():
    assert set(DEFAULT_SENTIMENT_SCORES) == set(DEFAULT_EMOTION_LABELS.values())


def test_normalize_emotion_labels_returns_copy_when_none():
    result = normalize_emotion_labels(None)
    assert result == DEFAULT_EMOTION_LABELS
    assert result is not DEFAULT_EMOTION_LABELS  # must be a copy, not the shared dict


def test_normalize_emotion_labels_coerces_keys_to_int():
    result = normalize_emotion_labels({"0": "Enjoyment", "1": "Sadness"})
    assert result == {0: "Enjoyment", 1: "Sadness"}


def test_normalize_sentiment_scores_coerces_values_to_float():
    result = normalize_sentiment_scores({"Enjoyment": 1})
    assert result["Enjoyment"] == 1.0
    assert isinstance(result["Enjoyment"], float)


def test_find_label_index_by_name_is_case_insensitive():
    assert find_label_index_by_name(DEFAULT_EMOTION_LABELS, "other") == 6
    assert find_label_index_by_name(DEFAULT_EMOTION_LABELS, "  ANGER ") == 2


def test_find_label_index_by_name_missing_returns_none():
    assert find_label_index_by_name(DEFAULT_EMOTION_LABELS, "Joy") is None
