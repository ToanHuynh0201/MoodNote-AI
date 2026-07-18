"""Tests for the synthetic-vs-real leakage detector (pure function, no I/O, no network)."""

import copy

from src.data.synthetic.leakage_guard import find_synthetic_leakage
from src.data.synthetic.schema import SyntheticSample, now_iso


def _make_sample(sample_id: str, text: str) -> SyntheticSample:
    return SyntheticSample(
        sample_id=sample_id,
        text=text,
        label=0,
        label_name="Enjoyment",
        model="Scripted-Test",
        channel="scripted",
        axis_style="a",
        axis_length="b",
        axis_context="c",
        prompt_template_id="diary_v1",
        generation_round=1,
        generated_at=now_iso(),
    )


def test_find_synthetic_leakage_detects_exact_match_against_test_split():
    samples = [_make_sample("s1", "Hôm nay tôi rất vui.")]
    real_texts = {"train": [], "validation": [], "test": ["hôm nay tôi rất vui."]}

    hits = find_synthetic_leakage(samples, real_texts, near_dup_threshold=90.0)

    assert hits["s1"].match_type == "exact"
    assert hits["s1"].matched_split == "test"
    assert hits["s1"].similarity == 100.0


def test_find_synthetic_leakage_detects_near_dup_against_train_split():
    samples = [_make_sample("s1", "hôm nay tôi rất vui vì đạt điểm cao trong kỳ thi")]
    real_texts = {
        "train": ["hôm nay tôi rất vui vì đạt điểm cao trong kỳ thi này"],
        "validation": [],
        "test": [],
    }

    hits = find_synthetic_leakage(samples, real_texts, near_dup_threshold=90.0)

    assert hits["s1"].match_type == "near"
    assert hits["s1"].matched_split == "train"


def test_find_synthetic_leakage_returns_empty_dict_when_clean():
    samples = [_make_sample("s1", "một câu nhật ký hoàn toàn không liên quan gì cả")]
    real_texts = {"train": ["chuyện khác hẳn"], "validation": [], "test": ["và một chuyện khác nữa"]}

    hits = find_synthetic_leakage(samples, real_texts, near_dup_threshold=90.0)

    assert hits == {}


def test_find_synthetic_leakage_does_not_mutate_input_samples():
    samples = [_make_sample("s1", "Hôm nay tôi rất vui.")]
    samples_before = copy.deepcopy(samples)
    real_texts = {"train": [], "validation": [], "test": ["hôm nay tôi rất vui."]}

    find_synthetic_leakage(samples, real_texts, near_dup_threshold=90.0)

    assert samples == samples_before
