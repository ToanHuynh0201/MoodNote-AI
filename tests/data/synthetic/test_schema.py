"""Tests for the SyntheticSample pydantic schema (no I/O, no network)."""

import pytest

from src.data.synthetic.schema import SyntheticSample, new_sample_id, now_iso


def _make_sample(**overrides) -> SyntheticSample:
    defaults = {
        "sample_id": "llama-3-8b-instruct-0-abc1234567",
        "text": "Hôm nay tôi rất vui vì đạt điểm cao.",
        "label": 0,
        "label_name": "Enjoyment",
        "model": "Llama-3-8B-Instruct",
        "channel": "scripted",
        "axis_style": "người lớn, điềm tĩnh, câu văn đầy đủ",
        "axis_length": "ngắn (2-3 câu)",
        "axis_context": "học tập, thi cử",
        "prompt_template_id": "diary_v1",
        "generation_round": 1,
        "generated_at": now_iso(),
    }
    defaults.update(overrides)
    return SyntheticSample(**defaults)


def test_synthetic_sample_round_trips_through_json():
    sample = _make_sample()

    restored = SyntheticSample.model_validate_json(sample.model_dump_json())

    assert restored == sample


def test_synthetic_sample_rejects_label_out_of_range():
    with pytest.raises(ValueError, match="label must be one of"):
        _make_sample(label=7)


def test_new_sample_id_produces_distinct_ids():
    ids = {new_sample_id("Llama-3-8B-Instruct", 0) for _ in range(20)}

    assert len(ids) == 20


def test_new_sample_id_slug_is_lowercase_with_dashes():
    sample_id = new_sample_id("Llama-3-8B-Instruct", 2)

    assert sample_id.startswith("llama-3-8b-instruct-2-")
