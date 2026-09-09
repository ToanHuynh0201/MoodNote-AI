"""
Gộp kết quả 3 kịch bản (đa seed) thành bảng so sánh cho báo cáo NCKH.

Đọc reports/ablation_<scenario>_seed<N>.json (do run_ablation.py sinh) → gộp seed thành
mean ± std → ghi reports/ablation_results.json + reports/ablation_comparison.md.
Nếu có đủ preds_<baseline>_seed<N>.csv và preds_combined_seed<N>.csv thì thêm kiểm định
ghép cặp McNemar (`combined` vs phương án nền, trên cùng tập test).

Cách dùng:
    python scripts/report_ablation.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.training.ablation_runner import (  # noqa: E402
    aggregate_seeds,
    compare_scenarios,
    render_comparison_markdown,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("report_ablation")


def _load_scenario_results(results_dir: Path, scenario: str) -> list[dict]:
    """Mọi file kết quả của một kịch bản: ưu tiên *_seed*.json, fallback file không seed."""
    files = sorted(results_dir.glob(f"ablation_{scenario}_seed*.json"))
    legacy = results_dir / f"ablation_{scenario}.json"
    if not files and legacy.exists():
        files = [legacy]
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def _read_preds(path: Path) -> list[tuple[int, int]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [(int(r["pred"]), int(r["true"])) for r in csv.DictReader(fh)]


def _mcnemar(base_preds: list[tuple[int, int]], treat_preds: list[tuple[int, int]]) -> dict:
    """McNemar exact (binomial) trên các cặp bất đồng đúng/sai giữa base và treat."""
    from scipy.stats import binomtest

    if len(base_preds) != len(treat_preds):
        raise ValueError(f"Số câu không khớp: base={len(base_preds)} vs treat={len(treat_preds)}")
    only_base = sum(
        1
        for (bp, bt), (tp, _) in zip(base_preds, treat_preds, strict=True)
        if bp == bt and tp != bt
    )
    only_treat = sum(
        1
        for (bp, bt), (tp, _) in zip(base_preds, treat_preds, strict=True)
        if bp != bt and tp == bt
    )
    discordant = only_base + only_treat
    p_value = 1.0 if discordant == 0 else binomtest(only_base, discordant, 0.5).pvalue
    return {
        "n": len(base_preds),
        "only_base_correct": only_base,
        "only_treat_correct": only_treat,
        "p_value": float(p_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ablation comparison report")
    parser.add_argument("--config-dir", default="configs", help="Thư mục chứa config")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    ablation_cfg = load_config(str(config_dir / "training_config.yaml"))["ablation"]
    results_dir = Path(ablation_cfg["results_dir"])

    per_scenario_runs: dict[str, list[dict]] = {}
    missing = []
    for scenario in ablation_cfg["scenarios"]:
        runs = _load_scenario_results(results_dir, scenario)
        if runs:
            per_scenario_runs[scenario] = runs
        else:
            missing.append(scenario)

    if missing:
        raise SystemExit(
            f"Thiếu kết quả của: {', '.join(missing)}. "
            f"Chạy `python scripts/run_ablation.py --scenario <tên> --seed <N>` trước."
        )

    aggregated = {name: aggregate_seeds(runs) for name, runs in per_scenario_runs.items()}

    comparison = compare_scenarios(
        aggregated,
        baseline=ablation_cfg["baseline"],
        metrics=tuple(ablation_cfg["metrics"]),
    )
    # Gắn std + số seed để render_comparison_markdown() hiện "mean ± std".
    comparison["n_seeds"] = {name: agg["n_seeds"] for name, agg in aggregated.items()}
    for name, agg in aggregated.items():
        if agg["n_seeds"] > 1:
            comparison["scenarios"][name]["scores_std"] = agg["metrics_std"]

    # McNemar: combined vs baseline trên seed đầu tiên có đủ cả 2 file preds.
    baseline = ablation_cfg["baseline"]
    for seed in ablation_cfg.get("seeds", [42]):
        base_file = results_dir / f"preds_{baseline}_seed{seed}.csv"
        treat_file = results_dir / f"preds_combined_seed{seed}.csv"
        if base_file.exists() and treat_file.exists():
            mcn = _mcnemar(_read_preds(base_file), _read_preds(treat_file))
            mcn["seed"] = seed
            comparison["mcnemar"] = mcn
            logger.info(
                f"McNemar (combined vs {baseline}, seed {seed}): p={mcn['p_value']:.4g} "
                f"(chỉ combined đúng={mcn['only_treat_correct']}, chỉ {baseline} đúng={mcn['only_base_correct']})"
            )
            break

    results_path = results_dir / "ablation_results.json"
    results_path.write_text(
        json.dumps(
            {"runs": per_scenario_runs, "aggregated": aggregated, "comparison": comparison},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    markdown_path = results_dir / "ablation_comparison.md"
    markdown_path.write_text(render_comparison_markdown(comparison), encoding="utf-8")

    logger.info(f"Đã ghi {results_path}")
    logger.info(f"Đã ghi {markdown_path}")
    logger.info(f"Kết luận: {'ĐẠT' if comparison['passed'] else 'CHƯA ĐẠT'}")


if __name__ == "__main__":
    main()
