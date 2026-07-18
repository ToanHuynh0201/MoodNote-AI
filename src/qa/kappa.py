"""
Cohen's Kappa giữa 2 người gán nhãn thủ công (sklearn.metrics.cohen_kappa_score —
dependency đã có sẵn trong requirements.txt).
"""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger("kappa")


def compute_cohens_kappa(rater_a_labels: list[int], rater_b_labels: list[int]) -> float | None:
    """
    Tính Cohen's Kappa giữa 2 người gán nhãn.

    Args:
        rater_a_labels: Nhãn (số) của người thứ nhất
        rater_b_labels: Nhãn (số) của người thứ hai, cùng thứ tự sample với rater_a_labels

    Returns:
        Giá trị Kappa, hoặc None (kèm log warning) nếu sklearn trả NaN — xảy ra khi 1
        người gán nhãn không có phương sai (toàn bộ cùng 1 nhãn, thường gặp ở fixture nhỏ).
        Trả None tường minh thay vì để NaN lan xuống các bước sau.
    """
    score = cohen_kappa_score(rater_a_labels, rater_b_labels)
    if math.isnan(score):
        logger.warning(
            "Cohen's Kappa không xác định (NaN) — một người gán nhãn không có phương sai."
        )
        return None
    return score


def compute_agreement_report(
    merged_labels_csv: str, qa_config_path: str = "configs/qa_config.yaml"
) -> dict:
    """
    Đọc merged_labels.csv, tính Kappa, so với ngưỡng qa_config.yaml's
    manual_audit.min_cohens_kappa.

    Args:
        merged_labels_csv: Đường dẫn file merged_labels.csv (sample_id, rater_a_label,
            rater_b_label) — do audit_sampling.import_rater_labels() sinh ra
        qa_config_path: Đường dẫn configs/qa_config.yaml

    Returns:
        {"n_samples": int, "kappa": float | None, "threshold": float, "passed": bool}
    """
    merged = pd.read_csv(merged_labels_csv, encoding="utf-8")
    qa_config = load_config(qa_config_path)
    threshold = qa_config["manual_audit"]["min_cohens_kappa"]

    kappa = compute_cohens_kappa(merged["rater_a_label"].tolist(), merged["rater_b_label"].tolist())
    passed = kappa is not None and kappa >= threshold

    report = {
        "n_samples": len(merged),
        "kappa": kappa,
        "threshold": threshold,
        "passed": passed,
    }
    logger.info(f"compute_agreement_report: {report}")
    return report
