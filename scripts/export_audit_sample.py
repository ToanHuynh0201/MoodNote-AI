"""
Rút mẫu audit thủ công và xuất sheet cho 2 người gán nhãn độc lập.

Cách dùng:
    python scripts/export_audit_sample.py --input data/synthetic/leakage_checked/clean.jsonl \
        --output-dir data/synthetic/qa/audit_sample
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.synthetic.schema import read_samples_jsonl  # noqa: E402
from src.qa.audit_sampling import draw_audit_sample, export_for_raters  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("export_audit_sample")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rút + xuất mẫu audit cho 2 người gán nhãn")
    parser.add_argument("--input", required=True, help="File JSONL đầu vào (đã qua leakage_guard)")
    parser.add_argument("--output-dir", default="data/synthetic/qa/audit_sample")
    parser.add_argument("--qa-config", default="configs/qa_config.yaml")
    args = parser.parse_args()

    qa_config = load_config(args.qa_config)
    manual_audit_cfg = qa_config["manual_audit"]

    samples = read_samples_jsonl(args.input)
    audit_sample = draw_audit_sample(
        samples, n=manual_audit_cfg["sample_size"], seed=manual_audit_cfg["seed"]
    )
    export_for_raters(
        audit_sample,
        output_dir=args.output_dir,
        rater_names=tuple(manual_audit_cfg["raters"]),
    )

    logger.info(
        f"Đã xuất {len(audit_sample)} mẫu audit vào {args.output_dir} — "
        f"gửi {manual_audit_cfg['raters']} điền tay cột 'label' rồi chạy compute_agreement.py."
    )


if __name__ == "__main__":
    main()
