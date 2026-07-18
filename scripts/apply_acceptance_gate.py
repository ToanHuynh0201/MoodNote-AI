"""
Áp cổng chấp nhận cuối cùng: lọc mẫu bị cross-LLM gắn cờ + đánh giá go/no-go theo đợt +
ghi data/synthetic/accepted/{train,validation,test}.csv.

Cách dùng:
    python scripts/apply_acceptance_gate.py \
        --clean-samples data/synthetic/leakage_checked/clean.jsonl \
        --cross-llm-reviews data/synthetic/qa/cross_llm/cross_llm_review.jsonl \
        --kappa-report data/synthetic/qa/audit_sample/kappa_report.json
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.qa.acceptance_gate import run_acceptance_gate  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("apply_acceptance_gate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Áp cổng chấp nhận dữ liệu synthetic")
    parser.add_argument("--clean-samples", required=True)
    parser.add_argument("--cross-llm-reviews", required=True)
    parser.add_argument("--kappa-report", required=True)
    parser.add_argument("--qa-config", default="configs/qa_config.yaml")
    parser.add_argument("--datagen-config", default="configs/datagen_config.yaml")
    parser.add_argument("--output-dir", default="data/synthetic/accepted")
    parser.add_argument("--round", type=int, default=1, dest="generation_round")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Dừng cứng (raise) nếu round verdict cần hiệu chỉnh prompt (mặc định chỉ log)",
    )
    args = parser.parse_args()

    verdict = run_acceptance_gate(
        clean_samples_path=args.clean_samples,
        cross_llm_reviews_path=args.cross_llm_reviews,
        kappa_report_path=args.kappa_report,
        qa_config_path=args.qa_config,
        datagen_config_path=args.datagen_config,
        output_dir=args.output_dir,
        generation_round=args.generation_round,
        strict=args.strict,
    )

    logger.info(f"Round verdict: {verdict.model_dump_json(indent=2)}")
    if verdict.needs_prompt_revision:
        logger.warning("Đợt này CẦN hiệu chỉnh prompt và sinh lại (xem 'reasons' ở trên).")
    else:
        logger.info("Đợt này đạt ngưỡng chất lượng.")


if __name__ == "__main__":
    main()
