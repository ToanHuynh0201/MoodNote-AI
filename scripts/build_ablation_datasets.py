"""
Ráp 3 tập train cho thí nghiệm ablation.

Cách dùng:
    python scripts/build_ablation_datasets.py
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.ablation.build_datasets import build_ablation_datasets  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("build_ablation_datasets_cli")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ablation train sets")
    parser.add_argument("--config-dir", default="configs", help="Thư mục chứa config")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    ablation_cfg = load_config(str(config_dir / "training_config.yaml"))["ablation"]

    counts = build_ablation_datasets(
        real_dir=ablation_cfg["real_dir"],
        accepted_dir=ablation_cfg["synthetic_dir"],
        output_dir=ablation_cfg["ablation_dir"],
        config_path=str(config_dir / "model_config.yaml"),
    )

    logger.info("Hoàn tất:")
    for name, n in counts.items():
        logger.info(f"  {name}: {n:,} mẫu")


if __name__ == "__main__":
    main()
