"""
Chạy data pipeline cho dữ liệu thật: download → preprocess (+ leakage-guard)

Cách dùng:
    python scripts/prepare_real_data.py
    python scripts/prepare_real_data.py --skip-download   # nếu đã có data/real/raw/
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("prepare_real_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real (UIT-VSMEC) training data")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Bỏ qua bước download (dùng khi đã có data/real/raw/)",
    )
    args = parser.parse_args()

    raw_dir = str(REPO_ROOT / "data" / "real" / "raw")
    processed_dir = str(REPO_ROOT / "data" / "real" / "processed")
    config_path = str(REPO_ROOT / "configs" / "model_config.yaml")

    if not args.skip_download:
        logger.info("Step 1/2: Downloading UIT-VSMEC...")
        from src.data.real.download_vsmec import download_uit_vsmec

        download_uit_vsmec(output_dir=raw_dir)
    else:
        logger.info("Skipping download (--skip-download)")

    logger.info("Step 2/2: Preprocessing (word segmentation + leakage guard)...")
    from src.data.real.preprocess import preprocess_dataset

    preprocess_dataset(
        input_dir=raw_dir,
        output_dir=processed_dir,
        config_path=config_path,
    )

    logger.info("Data pipeline hoàn tất!")
    logger.info(f"  Train : {processed_dir}/train.csv")
    logger.info(f"  Val   : {processed_dir}/validation.csv")
    logger.info(f"  Test  : {processed_dir}/test.csv")


if __name__ == "__main__":
    main()
