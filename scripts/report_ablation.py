"""
Gộp kết quả 3 kịch bản thành bảng so sánh cho báo cáo NCKH.

Đọc reports/ablation_<scenario>.json (do run_ablation.py sinh) rồi ghi
reports/ablation_results.json + reports/ablation_comparison.md.

Cách dùng:
    python scripts/report_ablation.py
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.training.ablation_runner import (  # noqa: E402
    compare_scenarios,
    render_comparison_markdown,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("report_ablation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ablation comparison report")
    parser.add_argument("--config-dir", default="configs", help="Thư mục chứa config")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    ablation_cfg = load_config(str(config_dir / "training_config.yaml"))["ablation"]
    results_dir = Path(ablation_cfg["results_dir"])

    results = {}
    missing = []
    for scenario in ablation_cfg["scenarios"]:
        result_file = results_dir / f"ablation_{scenario}.json"
        if result_file.exists():
            results[scenario] = json.loads(result_file.read_text(encoding="utf-8"))
        else:
            missing.append(scenario)

    if missing:
        raise SystemExit(
            f"Thiếu kết quả của: {', '.join(missing)}. "
            f"Chạy `python scripts/run_ablation.py --scenario <tên>` trước."
        )

    comparison = compare_scenarios(
        results,
        baseline=ablation_cfg["baseline"],
        metrics=tuple(ablation_cfg["metrics"]),
    )

    results_path = results_dir / "ablation_results.json"
    results_path.write_text(
        json.dumps({"results": results, "comparison": comparison}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown_path = results_dir / "ablation_comparison.md"
    markdown_path.write_text(render_comparison_markdown(comparison), encoding="utf-8")

    logger.info(f"Đã ghi {results_path}")
    logger.info(f"Đã ghi {markdown_path}")
    logger.info(f"Kết luận: {'ĐẠT' if comparison['passed'] else 'CHƯA ĐẠT'}")


if __name__ == "__main__":
    main()
