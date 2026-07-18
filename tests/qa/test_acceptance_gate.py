"""Tests for the per-sample filter, round verdict, and accepted/*.csv writer (no network)."""

import json

import pandas as pd
import pytest

from src.data.synthetic.schema import SyntheticSample, now_iso, write_samples_jsonl
from src.qa.acceptance_gate import (
    evaluate_round_verdict,
    filter_flagged_samples,
    run_acceptance_gate,
    split_train_val_test,
    write_accepted_csv,
)
from src.qa.cross_llm_check import CrossLLMFlag, CrossLLMReview, write_reviews_jsonl

_QA_CONFIG = {
    "manual_audit": {"min_cohens_kappa": 0.6},
    "cross_llm_audit": {"max_unnatural_rate": 0.1, "max_label_mismatch_rate": 0.15},
}


def _make_sample(sample_id: str, label: int = 0) -> SyntheticSample:
    return SyntheticSample(
        sample_id=sample_id,
        text=f"mẫu {sample_id}",
        label=label,
        label_name="Enjoyment" if label == 0 else "Sadness",
        model="Scripted-Test",
        channel="scripted",
        axis_style="a",
        axis_length="b",
        axis_context="c",
        prompt_template_id="diary_v1",
        generation_round=1,
        generated_at=now_iso(),
    )


def _make_samples_for_labels(labels_and_counts: dict[int, int]) -> list[SyntheticSample]:
    samples = []
    i = 0
    for label, count in labels_and_counts.items():
        for _ in range(count):
            samples.append(_make_sample(f"s{i}", label=label))
            i += 1
    return samples


def _make_review(sample_id: str, flag: CrossLLMFlag, reviewer_label: int = 0) -> CrossLLMReview:
    return CrossLLMReview(
        sample_id=sample_id, reviewer_model="Qwen3-8B", reviewer_label=reviewer_label, flag=flag
    )


def test_filter_flagged_samples_drops_only_flagged_when_enabled():
    samples = _make_samples_for_labels({0: 2})
    reviews = [_make_review(samples[0].sample_id, CrossLLMFlag.UNNATURAL_STYLE)]

    kept, dropped = filter_flagged_samples(samples, reviews, drop_cross_llm_flagged=True)

    assert [s.sample_id for s in dropped] == [samples[0].sample_id]
    assert [s.sample_id for s in kept] == [samples[1].sample_id]


def test_filter_flagged_samples_keeps_all_when_disabled():
    samples = _make_samples_for_labels({0: 2})
    reviews = [_make_review(samples[0].sample_id, CrossLLMFlag.UNNATURAL_STYLE)]

    kept, dropped = filter_flagged_samples(samples, reviews, drop_cross_llm_flagged=False)

    assert len(kept) == 2
    assert dropped == []


def test_split_train_val_test_respects_ratios_and_stratifies():
    samples = _make_samples_for_labels({0: 40, 1: 40})

    splits = split_train_val_test(
        samples, {"train": 0.7, "validation": 0.15, "test": 0.15}, seed=42
    )

    assert len(splits["train"]) + len(splits["validation"]) + len(splits["test"]) == 80
    assert len(splits["train"]) == pytest.approx(56, abs=2)
    assert len(splits["validation"]) == pytest.approx(12, abs=2)
    assert len(splits["test"]) == pytest.approx(12, abs=2)
    for split_samples in splits.values():
        assert {s.label for s in split_samples} == {0, 1}


def test_split_train_val_test_handles_too_few_samples_to_split_further():
    # Phát hiện qua dry-run tay thật (không phải chỉ unit test): với pool rất nhỏ, vòng
    # chia val/test thứ 2 có thể còn đúng 1 mẫu — sklearn không chia được 1 mẫu thành 2
    # tập con non-empty ở bất kỳ tỉ lệ nào.
    samples = _make_samples_for_labels({0: 3})

    splits = split_train_val_test(samples, {"train": 0.7, "validation": 0.15, "test": 0.15}, seed=1)

    assert len(splits["train"]) + len(splits["validation"]) + len(splits["test"]) == 3


def test_split_train_val_test_handles_single_sample():
    samples = _make_samples_for_labels({0: 1})

    splits = split_train_val_test(samples, {"train": 0.7, "validation": 0.15, "test": 0.15}, seed=1)

    assert splits == {"train": samples, "validation": [], "test": []}


def test_split_train_val_test_handles_empty_input():
    splits = split_train_val_test([], {"train": 0.7, "validation": 0.15, "test": 0.15}, seed=1)

    assert splits == {"train": [], "validation": [], "test": []}


def test_write_accepted_csv_output_compatible_with_real_preprocess_column_detection(tmp_path):
    from src.data.real.preprocess import _detect_label_column, _detect_text_column

    samples = _make_samples_for_labels({0: 2, 1: 1})
    splits = {"train": samples, "validation": [], "test": []}

    write_accepted_csv(splits, output_dir=str(tmp_path))

    df = pd.read_csv(tmp_path / "train.csv")
    text_col, used_fallback = _detect_text_column(list(df.columns))
    label_col = _detect_label_column(list(df.columns))

    assert text_col == "text"
    assert used_fallback is False
    assert label_col == "label"


def test_evaluate_round_verdict_passes_when_all_thresholds_met():
    reviews = [_make_review(f"s{i}", CrossLLMFlag.OK) for i in range(10)]

    verdict = evaluate_round_verdict(
        1, kappa_score=0.8, cross_llm_reviews=reviews, qa_config=_QA_CONFIG
    )

    assert verdict.needs_prompt_revision is False
    assert verdict.reasons == []


def test_evaluate_round_verdict_flags_low_kappa():
    reviews = [_make_review(f"s{i}", CrossLLMFlag.OK) for i in range(10)]

    verdict = evaluate_round_verdict(
        1, kappa_score=0.3, cross_llm_reviews=reviews, qa_config=_QA_CONFIG
    )

    assert verdict.needs_prompt_revision is True
    assert verdict.kappa_passed is False


def test_evaluate_round_verdict_flags_excess_unnatural_rate():
    reviews = [_make_review(f"s{i}", CrossLLMFlag.UNNATURAL_STYLE) for i in range(3)] + [
        _make_review(f"s{i}", CrossLLMFlag.OK) for i in range(3, 10)
    ]

    verdict = evaluate_round_verdict(
        1, kappa_score=0.8, cross_llm_reviews=reviews, qa_config=_QA_CONFIG
    )

    assert verdict.needs_prompt_revision is True
    assert verdict.unnatural_passed is False


def test_evaluate_round_verdict_treats_none_kappa_as_not_passed():
    reviews = [_make_review(f"s{i}", CrossLLMFlag.OK) for i in range(10)]

    verdict = evaluate_round_verdict(
        1, kappa_score=None, cross_llm_reviews=reviews, qa_config=_QA_CONFIG
    )

    assert verdict.kappa_passed is None
    assert verdict.needs_prompt_revision is True


def _write_run_acceptance_gate_fixtures(tmp_path, kappa: float | None):
    samples = _make_samples_for_labels({0: 10, 1: 10})
    clean_path = tmp_path / "clean.jsonl"
    write_samples_jsonl(samples, clean_path)

    reviews = [_make_review(samples[0].sample_id, CrossLLMFlag.UNNATURAL_STYLE)]
    reviews_path = tmp_path / "reviews.jsonl"
    write_reviews_jsonl(reviews, reviews_path)

    kappa_report_path = tmp_path / "kappa_report.json"
    kappa_report_path.write_text(json.dumps({"kappa": kappa}), encoding="utf-8")

    qa_config_path = tmp_path / "qa_config.yaml"
    qa_config_path.write_text(
        "manual_audit:\n  min_cohens_kappa: 0.6\n"
        "cross_llm_audit:\n  max_unnatural_rate: 0.1\n  max_label_mismatch_rate: 0.15\n"
        "acceptance:\n  drop_cross_llm_flagged: true\n",
        encoding="utf-8",
    )
    datagen_config_path = tmp_path / "datagen_config.yaml"
    datagen_config_path.write_text(
        "seed: 42\ngeneration:\n  split_ratios:\n    train: 0.7\n    validation: 0.15\n"
        "    test: 0.15\n",
        encoding="utf-8",
    )

    return {
        "clean_samples_path": str(clean_path),
        "cross_llm_reviews_path": str(reviews_path),
        "kappa_report_path": str(kappa_report_path),
        "qa_config_path": str(qa_config_path),
        "datagen_config_path": str(datagen_config_path),
    }


def test_run_acceptance_gate_always_writes_accepted_csv_even_when_verdict_fails(tmp_path):
    fixtures = _write_run_acceptance_gate_fixtures(tmp_path, kappa=0.1)
    output_dir = tmp_path / "accepted"

    verdict = run_acceptance_gate(
        **fixtures, output_dir=str(output_dir), generation_round=2, strict=False
    )

    assert verdict.needs_prompt_revision is True
    assert (output_dir / "train.csv").exists()
    assert (output_dir / "validation.csv").exists()
    assert (output_dir / "test.csv").exists()
    assert (tmp_path / "qa" / "round_verdict.json").exists()


def test_run_acceptance_gate_raises_when_strict_and_verdict_fails(tmp_path):
    fixtures = _write_run_acceptance_gate_fixtures(tmp_path, kappa=0.1)
    output_dir = tmp_path / "accepted"

    with pytest.raises(RuntimeError, match="chưa đạt ngưỡng"):
        run_acceptance_gate(**fixtures, output_dir=str(output_dir), generation_round=2, strict=True)


def test_run_acceptance_gate_passes_when_kappa_is_high(tmp_path):
    fixtures = _write_run_acceptance_gate_fixtures(tmp_path, kappa=0.95)
    output_dir = tmp_path / "accepted"

    verdict = run_acceptance_gate(
        **fixtures, output_dir=str(output_dir), generation_round=1, strict=False
    )

    # 1 mẫu bị cross-LLM gắn cờ trong fixture -> tỉ lệ unnatural = 1/1 = 100% > ngưỡng 10%,
    # nên verdict vẫn cần hiệu chỉnh dù Kappa cao — kiểm tra 2 tiêu chí độc lập nhau.
    assert verdict.kappa_passed is True
    assert verdict.unnatural_passed is False
