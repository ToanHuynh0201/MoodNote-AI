"""
Kiểm tra rò rỉ dữ liệu: mẫu synthetic nào trùng/near-duplicate với dữ liệu thật VSMEC.
So với data/real/raw (chưa tách từ) vì dữ liệu synthetic ở phase này cũng chưa tách từ
(việc tách từ pyvi để lại cho phase 4, sau khi QA xong).

Cách dùng:
    python scripts/check_leakage.py --input data/synthetic/dedup/deduped.jsonl \
        --real-dir data/real/raw --output-dir data/synthetic/leakage_checked
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.real.preprocess import _detect_text_column  # noqa: E402
from src.data.synthetic.leakage_guard import find_synthetic_leakage  # noqa: E402
from src.data.synthetic.schema import read_samples_jsonl, write_samples_jsonl  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("check_leakage")


def _load_real_texts(real_dir: str, splits: list[str]) -> dict[str, list[str]]:
    real_texts: dict[str, list[str]] = {}
    real_path = Path(real_dir)

    for split in splits:
        csv_path = real_path / f"{split}.csv"
        if not csv_path.exists():
            logger.warning(f"{csv_path} không tồn tại, bỏ qua split '{split}'.")
            continue
        df = pd.read_csv(csv_path)
        text_col, _ = _detect_text_column(list(df.columns))
        real_texts[split] = df[text_col].astype(str).tolist()

    return real_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra rò rỉ dữ liệu synthetic vs VSMEC thật")
    parser.add_argument("--input", required=True, help="File JSONL đầu vào (đã qua dedup)")
    parser.add_argument("--real-dir", default="data/real/raw", help="Thư mục chứa VSMEC gốc")
    parser.add_argument("--output-dir", default="data/synthetic/leakage_checked")
    parser.add_argument("--config", default="configs/datagen_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    guard_cfg = config["leakage_guard"]

    samples = read_samples_jsonl(args.input)
    real_texts = _load_real_texts(args.real_dir, guard_cfg["compare_splits"])

    hits = find_synthetic_leakage(
        samples, real_texts, near_dup_threshold=guard_cfg["near_dup_threshold"]
    )

    clean = [s for s in samples if s.sample_id not in hits]
    dropped = [s for s in samples if s.sample_id in hits]

    out_dir = Path(args.output_dir)
    write_samples_jsonl(clean, out_dir / "clean.jsonl")
    write_samples_jsonl(dropped, out_dir / "dropped_due_to_leakage.jsonl")

    report = {
        "n_input": len(samples),
        "n_clean": len(clean),
        "n_dropped": len(dropped),
        "hits": {
            sample_id: {
                "matched_split": hit.matched_split,
                "matched_text": hit.matched_text,
                "similarity": hit.similarity,
                "match_type": hit.match_type,
            }
            for sample_id, hit in hits.items()
        },
    }
    (out_dir / "leakage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        f"Hoàn tất: {len(samples)} -> {len(clean)} mẫu sạch (loại {len(dropped)} do rò rỉ)."
    )


if __name__ == "__main__":
    main()
