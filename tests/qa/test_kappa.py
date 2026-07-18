"""Tests for Cohen's Kappa computation (no network)."""

import pandas as pd
import pytest

from src.qa.kappa import compute_agreement_report, compute_cohens_kappa


def test_compute_cohens_kappa_matches_hand_computed_value():
    # 2 nhãn (0/1), 4 mẫu: 3 đồng thuận, 1 bất đồng -> po=0.75.
    # pe = sum(marginal_a * marginal_b) = (0.5*0.75)+(0.5*0.25) = 0.5
    # kappa = (po - pe) / (1 - pe) = (0.75 - 0.5) / 0.5 = 0.5
    rater_a = [0, 0, 1, 1]
    rater_b = [0, 0, 1, 0]

    kappa = compute_cohens_kappa(rater_a, rater_b)

    assert kappa == pytest.approx(0.5)


def test_compute_cohens_kappa_perfect_agreement_is_one():
    kappa = compute_cohens_kappa([0, 1, 2, 0, 1], [0, 1, 2, 0, 1])

    assert kappa == pytest.approx(1.0)


def test_compute_cohens_kappa_returns_none_for_zero_variance_rater(caplog):
    with caplog.at_level("WARNING"):
        kappa = compute_cohens_kappa([0, 0, 0], [0, 0, 0])

    assert kappa is None
    assert "không xác định" in caplog.text


def test_compute_agreement_report_passes_when_above_threshold(tmp_path):
    merged_path = tmp_path / "merged_labels.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "rater_a_label": [0, 1, 2, 3],
            "rater_b_label": [0, 1, 2, 3],
        }
    ).to_csv(merged_path, index=False)

    qa_config_path = tmp_path / "qa_config.yaml"
    qa_config_path.write_text("manual_audit:\n  min_cohens_kappa: 0.6\n", encoding="utf-8")

    report = compute_agreement_report(str(merged_path), qa_config_path=str(qa_config_path))

    assert report["n_samples"] == 4
    assert report["kappa"] == pytest.approx(1.0)
    assert report["passed"] is True


def test_compute_agreement_report_fails_when_below_threshold(tmp_path):
    merged_path = tmp_path / "merged_labels.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "rater_a_label": [0, 0, 1, 1],
            "rater_b_label": [0, 0, 1, 0],
        }
    ).to_csv(merged_path, index=False)

    qa_config_path = tmp_path / "qa_config.yaml"
    qa_config_path.write_text("manual_audit:\n  min_cohens_kappa: 0.9\n", encoding="utf-8")

    report = compute_agreement_report(str(merged_path), qa_config_path=str(qa_config_path))

    assert report["passed"] is False
