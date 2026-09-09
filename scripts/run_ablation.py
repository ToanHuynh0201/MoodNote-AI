"""
Chạy thí nghiệm ablation: fine-tune PhoBERT cho 1 hoặc cả 3 kịch bản.

Mỗi kịch bản ghi reports/ablation_<scenario>.json riêng, nên chạy được từng kịch bản
một — cần thiết vì phiên Colab miễn phí có thể ngắt giữa chừng.

Cách dùng:
    python scripts/run_ablation.py --scenario real_only
    python scripts/run_ablation.py --scenario all --no-wandb
    python scripts/run_ablation.py --scenario all --smoke     # thử đường ống trên CPU
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.training.ablation_runner import SCENARIOS, run_scenario  # noqa: E402
from src.utils.config import load_all_configs  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("run_ablation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PhoBERT ablation scenarios")
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS, "all"],
        required=True,
        help="Kịch bản cần chạy, hoặc 'all' để chạy tuần tự cả 3",
    )
    parser.add_argument("--config-dir", default="configs", help="Thư mục chứa config")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed cho lần chạy này (mặc định: training.seed trong config). "
        "File kết quả gắn hậu tố _seed<N> nên chạy nhiều seed không đè nhau.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Tắt logging W&B")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Chạy thử rút gọn (32 dòng/split, 1 epoch) để kiểm tra đường ống. Tự tắt W&B.",
    )
    args = parser.parse_args()

    configs = load_all_configs(args.config_dir)
    model_config = configs["model"]
    training_config = configs["training"]
    ablation_cfg = training_config["ablation"]

    use_wandb = (
        not args.no_wandb
        and not args.smoke
        and training_config.get("wandb", {}).get("enabled", True)
    )

    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        logger.info("=" * 60)
        logger.info(f"Kịch bản: {scenario}")
        logger.info("=" * 60)
        result = run_scenario(
            scenario=scenario,
            model_config=model_config,
            training_config=training_config,
            ablation_dir=ablation_cfg["ablation_dir"],
            validation_path=ablation_cfg["validation_path"],
            test_path=ablation_cfg["test_path"],
            results_dir=ablation_cfg["results_dir"],
            use_wandb=use_wandb,
            smoke=args.smoke,
            seed=args.seed,
        )
        m = result["metrics"]
        logger.info(
            f"{scenario} (seed {result['seed']}): accuracy={m['accuracy']:.4f} "
            f"f1_macro={m['f1_macro']:.4f} f1_weighted={m['f1_weighted']:.4f}"
        )

    logger.info("Xong. Chạy scripts/report_ablation.py để sinh bảng so sánh.")


if __name__ == "__main__":
    main()
