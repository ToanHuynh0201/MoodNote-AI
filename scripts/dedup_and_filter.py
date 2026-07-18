"""
Loại bỏ trùng lặp exact-hash + near-duplicate trong 1 (hoặc nhiều) file JSONL đã sinh.

Cách dùng:
    python scripts/dedup_and_filter.py --input data/synthetic/raw/llama3_round1.jsonl \
        --output-dir data/synthetic/dedup
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.synthetic.dedup import exact_dedup, near_dedup  # noqa: E402
from src.data.synthetic.schema import read_samples_jsonl, write_samples_jsonl  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("dedup_and_filter")


def main() -> None:
    parser = argparse.ArgumentParser(description="Loại bỏ trùng lặp trong dữ liệu synthetic")
    parser.add_argument("--input", required=True, nargs="+", help="1 hoặc nhiều file JSONL đầu vào")
    parser.add_argument("--output-dir", default="data/synthetic/dedup")
    parser.add_argument("--config", default="configs/datagen_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    threshold = config["dedup"]["near_dup_threshold"]

    samples = []
    for input_path in args.input:
        samples.extend(read_samples_jsonl(input_path))
    logger.info(f"Đã đọc {len(samples)} mẫu từ {len(args.input)} file.")

    after_exact, n_exact_removed = exact_dedup(samples)
    after_near, near_dropped_report = near_dedup(after_exact, threshold=threshold)

    out_dir = Path(args.output_dir)
    write_samples_jsonl(after_near, out_dir / "deduped.jsonl")

    report = {
        "n_input": len(samples),
        "n_exact_removed": n_exact_removed,
        "n_near_removed": len(near_dropped_report),
        "n_kept": len(after_near),
        "near_dup_threshold": threshold,
        "near_dup_dropped": near_dropped_report,
    }
    (out_dir / "dedup_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        f"Hoàn tất: {len(samples)} -> {len(after_near)} mẫu "
        f"(loại {n_exact_removed} exact + {len(near_dropped_report)} near-dup)."
    )


if __name__ == "__main__":
    main()
