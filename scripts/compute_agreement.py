"""
Ghép nhãn của 2 người audit + tính Cohen's Kappa.

Cách dùng:
    python scripts/compute_agreement.py \
        --rater-a data/synthetic/qa/audit_sample/rater_a_sheet.csv \
        --rater-b data/synthetic/qa/audit_sample/rater_b_sheet.csv
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.qa.audit_sampling import import_rater_labels  # noqa: E402
from src.qa.kappa import compute_agreement_report  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("compute_agreement")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tính Cohen's Kappa giữa 2 người audit")
    parser.add_argument("--rater-a", required=True, help="Sheet đã điền của người thứ nhất")
    parser.add_argument("--rater-b", required=True, help="Sheet đã điền của người thứ hai")
    parser.add_argument("--qa-config", default="configs/qa_config.yaml")
    args = parser.parse_args()

    merged = import_rater_labels(args.rater_a, args.rater_b)
    merged_path = Path(args.rater_a).parent / "merged_labels.csv"

    report = compute_agreement_report(str(merged_path), qa_config_path=args.qa_config)

    kappa_report_path = Path(args.rater_a).parent / "kappa_report.json"
    kappa_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"Đã ghép {len(merged)} mẫu. Kappa report ghi tại {kappa_report_path}: {report}")


if __name__ == "__main__":
    main()
