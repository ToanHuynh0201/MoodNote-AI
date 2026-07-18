"""
Cổng chấp nhận dữ liệu synthetic: lọc từng mẫu bị cross-LLM gắn cờ + đánh giá go/no-go
theo cả đợt sinh (Cohen's Kappa + tỉ lệ unnatural/mismatch), rồi ghi
data/synthetic/accepted/{train,validation,test}.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from ..data.synthetic.schema import SyntheticSample, read_samples_jsonl
from ..utils.config import load_config
from ..utils.logger import get_logger
from .cross_llm_check import CrossLLMFlag, CrossLLMReview, read_reviews_jsonl

logger = get_logger("acceptance_gate")


class RoundVerdict(BaseModel):
    generation_round: int
    kappa_score: float | None
    kappa_passed: bool | None
    unnatural_rate: float
    unnatural_passed: bool
    label_mismatch_rate: float
    label_mismatch_passed: bool
    needs_prompt_revision: bool
    reasons: list[str]


def evaluate_round_verdict(
    generation_round: int,
    kappa_score: float | None,
    cross_llm_reviews: list[CrossLLMReview],
    qa_config: dict,
) -> RoundVerdict:
    """
    Đánh giá tín hiệu go/no-go cho cả đợt sinh dữ liệu (đúng vòng lặp "audit -> chỉnh
    prompt -> sinh lại" thuyết minh mô tả). Hàm này KHÔNG tự sửa prompt — chỉ báo hiệu
    cho người quyết định.

    Args:
        generation_round: Số thứ tự đợt sinh/hiệu chỉnh prompt
        kappa_score: Kết quả Cohen's Kappa (None nếu không xác định được)
        cross_llm_reviews: Kết quả cross-LLM audit của đợt này
        qa_config: Nội dung configs/qa_config.yaml đã load

    Returns:
        RoundVerdict với cờ needs_prompt_revision + lý do cụ thể
    """
    kappa_threshold = qa_config["manual_audit"]["min_cohens_kappa"]
    kappa_passed = None if kappa_score is None else kappa_score >= kappa_threshold

    n_reviews = len(cross_llm_reviews)
    unnatural_rate = (
        sum(1 for r in cross_llm_reviews if r.flag == CrossLLMFlag.UNNATURAL_STYLE) / n_reviews
        if n_reviews
        else 0.0
    )
    label_mismatch_rate = (
        sum(1 for r in cross_llm_reviews if r.flag == CrossLLMFlag.LABEL_MISMATCH) / n_reviews
        if n_reviews
        else 0.0
    )

    max_unnatural_rate = qa_config["cross_llm_audit"]["max_unnatural_rate"]
    max_label_mismatch_rate = qa_config["cross_llm_audit"]["max_label_mismatch_rate"]
    unnatural_passed = unnatural_rate <= max_unnatural_rate
    label_mismatch_passed = label_mismatch_rate <= max_label_mismatch_rate

    reasons = []
    if kappa_passed is not True:
        reasons.append(
            f"Cohen's Kappa {kappa_score} chưa đạt ngưỡng {kappa_threshold} (hoặc không xác định)"
        )
    if not unnatural_passed:
        reasons.append(
            f"Tỉ lệ mẫu bị gắn cờ 'rập khuôn' {unnatural_rate:.2%} vượt ngưỡng "
            f"{max_unnatural_rate:.2%}"
        )
    if not label_mismatch_passed:
        reasons.append(
            f"Tỉ lệ cross-LLM bất đồng nhãn {label_mismatch_rate:.2%} vượt ngưỡng "
            f"{max_label_mismatch_rate:.2%}"
        )

    return RoundVerdict(
        generation_round=generation_round,
        kappa_score=kappa_score,
        kappa_passed=kappa_passed,
        unnatural_rate=unnatural_rate,
        unnatural_passed=unnatural_passed,
        label_mismatch_rate=label_mismatch_rate,
        label_mismatch_passed=label_mismatch_passed,
        needs_prompt_revision=bool(reasons),
        reasons=reasons,
    )


def filter_flagged_samples(
    samples: list[SyntheticSample],
    cross_llm_reviews: list[CrossLLMReview],
    drop_cross_llm_flagged: bool = True,
) -> tuple[list[SyntheticSample], list[SyntheticSample]]:
    """
    Lọc từng mẫu bị cross-LLM gắn cờ, độc lập với round verdict.

    Args:
        samples: Pool mẫu cần lọc
        cross_llm_reviews: Kết quả cross-LLM audit
        drop_cross_llm_flagged: Nếu False, không lọc gì cả (giữ nguyên toàn bộ)

    Returns:
        (kept, dropped)
    """
    if not drop_cross_llm_flagged:
        return samples, []

    flagged_ids = {r.sample_id for r in cross_llm_reviews if r.flag != CrossLLMFlag.OK}
    kept = [s for s in samples if s.sample_id not in flagged_ids]
    dropped = [s for s in samples if s.sample_id in flagged_ids]

    if dropped:
        logger.info(f"filter_flagged_samples: loại {len(dropped)} mẫu bị cross-LLM gắn cờ.")

    return kept, dropped


def split_train_val_test(
    samples: list[SyntheticSample], ratios: dict[str, float], seed: int
) -> dict[str, list[SyntheticSample]]:
    """
    Chia train/validation/test theo tỉ lệ cấu hình, stratified theo nhãn khi có thể.

    Args:
        samples: Pool mẫu đã lọc
        ratios: {"train": ..., "validation": ..., "test": ...} (tổng = 1.0)
        seed: Seed cho việc chia tách

    Returns:
        {"train": [...], "validation": [...], "test": [...]}
    """
    if len(samples) < 2:
        # sklearn không thể chia 0-1 mẫu thành train/test non-empty ở bất kỳ tỉ lệ nào.
        return {"train": list(samples), "validation": [], "test": []}

    labels = [s.label for s in samples]
    val_test_size = ratios["validation"] + ratios["test"]

    try:
        train_samples, temp_samples, _, temp_labels = train_test_split(
            samples, labels, test_size=val_test_size, random_state=seed, stratify=labels
        )
    except ValueError as e:
        logger.warning(f"split_train_val_test: stratify thất bại ({e}), dùng split không stratify.")
        train_samples, temp_samples = train_test_split(
            samples, test_size=val_test_size, random_state=seed
        )
        temp_labels = [s.label for s in temp_samples]

    if len(temp_samples) < 2:
        # Tương tự — không đủ mẫu để chia tiếp val/test, dồn hết vào validation.
        return {"train": train_samples, "validation": temp_samples, "test": []}

    test_fraction_of_temp = ratios["test"] / val_test_size
    try:
        val_samples, test_samples = train_test_split(
            temp_samples, test_size=test_fraction_of_temp, random_state=seed, stratify=temp_labels
        )
    except ValueError:
        val_samples, test_samples = train_test_split(
            temp_samples, test_size=test_fraction_of_temp, random_state=seed
        )

    return {"train": train_samples, "validation": val_samples, "test": test_samples}


def write_accepted_csv(
    splits: dict[str, list[SyntheticSample]], output_dir: str = "data/synthetic/accepted"
) -> None:
    """
    Ghi accepted/{train,validation,test}.csv. Cột `text`/`label` khớp chính xác với
    `_detect_text_column`/`_detect_label_column` của `src/data/real/preprocess.py`.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_samples in splits.items():
        df = pd.DataFrame(
            {
                "sample_id": [s.sample_id for s in split_samples],
                "text": [s.text for s in split_samples],
                "label": [s.label for s in split_samples],
                "model": [s.model for s in split_samples],
                "generation_round": [s.generation_round for s in split_samples],
            }
        )
        df.to_csv(out_dir / f"{split_name}.csv", index=False, encoding="utf-8")

    total = sum(len(v) for v in splits.values())
    logger.info(f"write_accepted_csv: đã ghi {total} mẫu vào {out_dir}")


def run_acceptance_gate(
    clean_samples_path: str,
    cross_llm_reviews_path: str,
    kappa_report_path: str,
    qa_config_path: str = "configs/qa_config.yaml",
    datagen_config_path: str = "configs/datagen_config.yaml",
    output_dir: str = "data/synthetic/accepted",
    generation_round: int = 1,
    strict: bool = False,
) -> RoundVerdict:
    """
    Orchestrator: luôn ghi accepted/*.csv từ pool đã lọc từng-mẫu, luôn ghi
    round_verdict.json. Không tự dừng cứng khi round verdict fail trừ khi strict=True.

    Args:
        clean_samples_path: JSONL các SyntheticSample đã qua dedup + leakage_guard
        cross_llm_reviews_path: JSONL các CrossLLMReview
        kappa_report_path: JSON report từ src/qa/kappa.py's compute_agreement_report()
        qa_config_path: Đường dẫn configs/qa_config.yaml
        datagen_config_path: Đường dẫn configs/datagen_config.yaml (seed + split_ratios)
        output_dir: Thư mục ghi accepted/*.csv
        generation_round: Số thứ tự đợt sinh, lưu vào round_verdict.json
        strict: Nếu True và needs_prompt_revision=True, raise thay vì chỉ log

    Returns:
        RoundVerdict của đợt này

    Raises:
        RuntimeError: nếu strict=True và verdict.needs_prompt_revision là True
    """
    qa_config = load_config(qa_config_path)
    datagen_config = load_config(datagen_config_path)

    samples = read_samples_jsonl(clean_samples_path)
    cross_llm_reviews = read_reviews_jsonl(cross_llm_reviews_path)

    with open(kappa_report_path, encoding="utf-8") as f:
        kappa_report = json.load(f)

    verdict = evaluate_round_verdict(
        generation_round, kappa_report["kappa"], cross_llm_reviews, qa_config
    )

    kept, _dropped = filter_flagged_samples(
        samples, cross_llm_reviews, qa_config["acceptance"]["drop_cross_llm_flagged"]
    )

    splits = split_train_val_test(
        kept, datagen_config["generation"]["split_ratios"], seed=datagen_config["seed"]
    )
    write_accepted_csv(splits, output_dir=output_dir)

    verdict_path = Path(output_dir).parent / "qa" / "round_verdict.json"
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_path.write_text(verdict.model_dump_json(indent=2), encoding="utf-8")

    if strict and verdict.needs_prompt_revision:
        raise RuntimeError(
            f"run_acceptance_gate: round {generation_round} chưa đạt ngưỡng chất lượng: "
            f"{verdict.reasons}"
        )

    return verdict
