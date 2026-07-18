"""Tests for manual-audit sampling + blind 2-rater export/import (no network)."""

import pandas as pd
import pytest

from src.data.synthetic.schema import SyntheticSample, now_iso
from src.qa.audit_sampling import draw_audit_sample, export_for_raters, import_rater_labels


def _make_samples(n: int) -> list[SyntheticSample]:
    return [
        SyntheticSample(
            sample_id=f"s{i}",
            text=f"mẫu nhật ký số {i}",
            label=i % 7,
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
        for i in range(n)
    ]


def test_draw_audit_sample_is_deterministic_with_seed():
    samples = _make_samples(20)

    first = draw_audit_sample(samples, n=5, seed=7)
    second = draw_audit_sample(samples, n=5, seed=7)

    assert [s.sample_id for s in first] == [s.sample_id for s in second]
    assert len(first) == 5


def test_draw_audit_sample_takes_all_when_n_exceeds_pool_size(caplog):
    samples = _make_samples(3)

    with caplog.at_level("WARNING"):
        result = draw_audit_sample(samples, n=10, seed=1)

    assert len(result) == 3
    assert "chỉ có 3" in caplog.text


def test_export_for_raters_writes_matching_blank_sheets(tmp_path):
    samples = _make_samples(3)

    export_for_raters(samples, output_dir=str(tmp_path), rater_names=("rater_a", "rater_b"))

    rater_a = pd.read_csv(tmp_path / "rater_a_sheet.csv")
    rater_b = pd.read_csv(tmp_path / "rater_b_sheet.csv")

    assert (
        list(rater_a["sample_id"]) == list(rater_b["sample_id"]) == [s.sample_id for s in samples]
    )
    assert rater_a["label"].isna().all() or (rater_a["label"] == "").all()


def test_export_for_raters_sheets_never_contain_model_generated_label(tmp_path):
    samples = _make_samples(2)

    export_for_raters(samples, output_dir=str(tmp_path), rater_names=("rater_a", "rater_b"))

    rater_a = pd.read_csv(tmp_path / "rater_a_sheet.csv")
    rater_b = pd.read_csv(tmp_path / "rater_b_sheet.csv")

    assert "model_generated_label" not in rater_a.columns
    assert "model_generated_label" not in rater_b.columns
    blind_pool = pd.read_csv(tmp_path / "blind_pool.csv")
    assert "model_generated_label" in blind_pool.columns


def _fill_sheet(path, sample_ids, labels):
    pd.DataFrame(
        {"sample_id": sample_ids, "text": ["x"] * len(sample_ids), "label": labels}
    ).to_csv(path, index=False, encoding="utf-8")


def test_import_rater_labels_converts_names_case_insensitively(tmp_path):
    rater_a_path = tmp_path / "rater_a_sheet.csv"
    rater_b_path = tmp_path / "rater_b_sheet.csv"
    _fill_sheet(rater_a_path, ["s0", "s1"], ["enjoyment", "SADNESS"])
    _fill_sheet(rater_b_path, ["s0", "s1"], ["Enjoyment", "sadness"])

    merged = import_rater_labels(str(rater_a_path), str(rater_b_path))

    assert merged["rater_a_label"].tolist() == [0, 1]
    assert merged["rater_b_label"].tolist() == [0, 1]
    assert (tmp_path / "merged_labels.csv").exists()


def test_import_rater_labels_raises_on_unknown_label_name(tmp_path):
    rater_a_path = tmp_path / "rater_a_sheet.csv"
    rater_b_path = tmp_path / "rater_b_sheet.csv"
    _fill_sheet(rater_a_path, ["s0"], ["Enjoyment"])
    _fill_sheet(rater_b_path, ["s0"], ["Enjoymentt"])  # typo

    with pytest.raises(ValueError, match="tên nhãn không hợp lệ"):
        import_rater_labels(str(rater_a_path), str(rater_b_path))


def test_import_rater_labels_raises_on_mismatched_sample_ids(tmp_path):
    rater_a_path = tmp_path / "rater_a_sheet.csv"
    rater_b_path = tmp_path / "rater_b_sheet.csv"
    _fill_sheet(rater_a_path, ["s0", "s1"], ["Enjoyment", "Sadness"])
    _fill_sheet(rater_b_path, ["s0", "s2"], ["Enjoyment", "Sadness"])

    with pytest.raises(ValueError, match="lệch tập sample_id"):
        import_rater_labels(str(rater_a_path), str(rater_b_path))
