"""
Thí nghiệm ablation 3 kịch bản (Nội dung 2 của thuyết minh).

Ba kịch bản `real_only` / `synthetic_only` / `combined` dùng CHUNG một bộ
hyperparameter, CHUNG tập validation và CHUNG tập test cố định của UIT-VSMEC —
biến duy nhất là tập train.

Hai hàm `compare_scenarios()` và `render_comparison_markdown()` là hàm thuần, không
đụng tới torch, nên import được ở môi trường không cài GPU stack.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger

logger = get_logger("ablation_runner")

SCENARIOS = ("real_only", "synthetic_only", "combined")
DEFAULT_METRICS = ("accuracy", "f1_macro", "f1_weighted")
SMOKE_ROWS = 32


def _smoke_csv(src: str | Path, out_dir: str | Path, n_rows: int = SMOKE_ROWS) -> str:
    """Cắt một CSV còn n_rows dòng đầu, ghi ra out_dir và trả về đường dẫn mới."""
    import pandas as pd

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dst = out_path / Path(src).name
    pd.read_csv(src).head(n_rows).to_csv(dst, index=False, encoding="utf-8")
    return str(dst)


def _smoke_training_config(training_config: dict) -> dict:
    """Bản sao config rút gọn để chạy thử toàn tuyến trên CPU trong vài phút."""
    cfg = copy.deepcopy(training_config)
    cfg["training"]["num_epochs"] = 1
    cfg["training"]["early_stopping_patience"] = 1
    cfg["logging"]["log_steps"] = 1
    cfg["logging"]["eval_steps"] = 1
    cfg["logging"]["save_steps"] = 1
    return cfg


def _compute_class_weights(train_labels: list[int], num_labels: int):
    """
    Class weights 'balanced' tính trên chính tập train của kịch bản đó.

    Lớp không xuất hiện trong tập train (chỉ xảy ra ở chế độ --smoke) nhận trọng số 1.0
    thay vì làm sklearn ném lỗi.
    """
    import numpy as np
    import torch
    from sklearn.utils.class_weight import compute_class_weight

    present = np.unique(train_labels)
    weights = np.ones(num_labels, dtype=np.float32)
    weights[present] = compute_class_weight("balanced", classes=present, y=train_labels)
    return torch.tensor(weights, dtype=torch.float32)


def run_scenario(
    scenario: str,
    model_config: dict,
    training_config: dict,
    ablation_dir: str = "data/ablation",
    validation_path: str = "data/real/processed/validation.csv",
    test_path: str = "data/real/processed/test.csv",
    results_dir: str = "reports",
    use_wandb: bool = True,
    smoke: bool = False,
) -> dict[str, Any]:
    """
    Fine-tune PhoBERT cho một kịch bản rồi đánh giá trên tập test cố định.

    Args:
        scenario: Tên kịch bản (phải nằm trong SCENARIOS)
        model_config: Nội dung configs/model_config.yaml đã load
        training_config: Nội dung configs/training_config.yaml đã load
        ablation_dir: Thư mục chứa <scenario>/train.csv do build_datasets.py sinh ra
        validation_path: Tập validation dùng chung cho cả 3 kịch bản
        test_path: Tập test cố định của UIT-VSMEC — chỉ đọc, không bao giờ ghi
        results_dir: Nơi ghi ablation_<scenario>.json và confusion matrix
        use_wandb: Bật/tắt logging W&B
        smoke: Chạy thử rút gọn (32 dòng mỗi split, 1 epoch) để kiểm tra đường ống

    Returns:
        dict kết quả của kịch bản (đồng thời được ghi ra reports/ablation_<scenario>.json)
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Kịch bản không hợp lệ: {scenario!r}. Chọn trong {SCENARIOS}.")

    # Import nặng để trong hàm: compare_scenarios/render_comparison_markdown phải
    # dùng được ở môi trường không cài torch/transformers.
    from ..data.real.dataset import EmotionDataset
    from ..models.model_utils import get_device, print_model_summary, save_model
    from ..models.phobert_classifier import create_model
    from ..utils.metrics import compute_metrics, plot_confusion_matrix, print_metrics
    from .trainer import train_model

    train_path = str(Path(ablation_dir) / scenario / "train.csv")
    if smoke:
        smoke_dir = Path(ablation_dir) / "_smoke" / scenario
        train_path = _smoke_csv(train_path, smoke_dir)
        validation_path = _smoke_csv(validation_path, smoke_dir)
        test_path = _smoke_csv(test_path, smoke_dir)
        training_config = _smoke_training_config(training_config)
        logger.warning(f"[{scenario}] CHẾ ĐỘ SMOKE — số liệu không dùng cho báo cáo")

    m_cfg = model_config["model"]
    logger.info(f"[{scenario}] Nạp dữ liệu...")
    datasets = {}
    tokenizer = None
    for split, path in (
        ("train", train_path),
        ("validation", validation_path),
        ("test", test_path),
    ):
        datasets[split] = EmotionDataset(
            data_path=path,
            tokenizer_name=m_cfg["name"],
            max_length=m_cfg["max_seq_length"],
            tokenizer=tokenizer,
        )
        tokenizer = datasets[split].tokenizer
    logger.info(
        f"[{scenario}] train={len(datasets['train'])} "
        f"validation={len(datasets['validation'])} test={len(datasets['test'])}"
    )

    class_weights = None
    if training_config["training"].get("use_class_weights", True):
        class_weights = _compute_class_weights(datasets["train"].labels, m_cfg["num_labels"])
        logger.info(f"[{scenario}] class weights: {class_weights.tolist()}")

    model = create_model(model_config, class_weights=class_weights)
    model.to(get_device())
    print_model_summary(model)

    # Mỗi kịch bản là một run W&B riêng để so sánh được trên cùng project.
    run_config = copy.deepcopy(training_config)
    wandb_cfg = run_config.setdefault("wandb", {})
    wandb_cfg["name"] = f"{wandb_cfg.get('name', 'phobert')}-{scenario}"

    trainer = train_model(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        training_config=run_config,
        output_dir=f"models/checkpoints/{scenario}",
        use_wandb=use_wandb,
    )

    logger.info(f"[{scenario}] Đánh giá trên tập test cố định...")
    predictions = trainer.predict(datasets["test"])
    metrics = compute_metrics(predictions.predictions, predictions.label_ids)
    print_metrics(metrics, model_config["emotion_labels"])

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        predictions.predictions,
        predictions.label_ids,
        emotion_labels=model_config["emotion_labels"],
        save_path=results_path / f"confusion_matrix_{scenario}.png",
    )

    model_dir = f"models/best_model/{scenario}"
    save_model(
        model=trainer.model,
        tokenizer=tokenizer,
        save_dir=model_dir,
        config={"model_config": model_config, "training_config": run_config},
    )

    result = {
        "scenario": scenario,
        "smoke": smoke,
        "model_name": m_cfg["name"],
        "seed": training_config["training"]["seed"],
        "num_epochs": training_config["training"]["num_epochs"],
        "train_csv": train_path,
        "validation_csv": validation_path,
        "test_csv": test_path,
        "n_train": len(datasets["train"]),
        "n_validation": len(datasets["validation"]),
        "n_test": len(datasets["test"]),
        "model_dir": model_dir,
        "metrics": metrics,
    }

    out_file = results_path / f"ablation_{scenario}.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[{scenario}] Đã ghi kết quả -> {out_file}")

    return result


def compare_scenarios(
    results: dict[str, dict],
    baseline: str = "real_only",
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[str, Any]:
    """
    So sánh các kịch bản với phương án nền — hàm thuần, không phụ thuộc torch.

    Tiêu chí pass/fail bám đúng thuyết minh ("đạt độ chính xác cao hơn so với phương án
    nền"): `combined` phải vượt `baseline` trên cả 3 chỉ số. KHÔNG dùng ngưỡng tuyệt đối
    nào — thuyết minh không quy định con số nào.

    Args:
        results: {tên kịch bản: dict kết quả của run_scenario}
        baseline: Kịch bản đóng vai phương án nền
        metrics: Các chỉ số so sánh

    Returns:
        dict so sánh, đầu vào cho render_comparison_markdown()
    """
    if baseline not in results:
        raise ValueError(f"Thiếu kết quả của phương án nền {baseline!r}: có {sorted(results)}")

    base_metrics = results[baseline]["metrics"]

    rows = {}
    for name, result in results.items():
        scores = {m: result["metrics"][m] for m in metrics}
        rows[name] = {
            "n_train": result["n_train"],
            "scores": scores,
            "delta_vs_baseline": {m: scores[m] - base_metrics[m] for m in metrics},
        }

    contribution = rows["combined"]["delta_vs_baseline"] if "combined" in rows else {}
    failed = [m for m, d in contribution.items() if d <= 0]

    return {
        "baseline": baseline,
        "metrics": list(metrics),
        "scenarios": rows,
        "synthetic_contribution": contribution,
        "passed": bool(contribution) and not failed,
        "failed_metrics": failed,
        "smoke": any(r.get("smoke") for r in results.values()),
    }


def render_comparison_markdown(comparison: dict) -> str:
    """
    Sinh reports/ablation_comparison.md từ kết quả compare_scenarios() — hàm thuần.

    Args:
        comparison: Đầu ra của compare_scenarios()

    Returns:
        Nội dung markdown
    """
    metrics = comparison["metrics"]
    baseline = comparison["baseline"]

    lines = [
        "# Ablation 3 kịch bản — PhoBERT phân tích cảm xúc tiếng Việt",
        "",
        "Ba kịch bản dùng chung một bộ hyperparameter, chung tập validation và chung",
        "tập test cố định của UIT-VSMEC. Biến duy nhất là tập dữ liệu huấn luyện.",
        "",
    ]

    if comparison["smoke"]:
        lines += [
            "> ⚠️ **CHẠY SMOKE** — dữ liệu bị cắt còn 32 dòng, 1 epoch.",
            "> Số liệu dưới đây chỉ để kiểm tra đường ống, KHÔNG dùng cho báo cáo.",
            "",
        ]

    header = ["Kịch bản", "Số mẫu train", *metrics]
    lines += [
        "## Kết quả trên tập test cố định",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for name, row in comparison["scenarios"].items():
        tag = f"`{name}`" + (" (phương án nền)" if name == baseline else "")
        cells = [tag, f"{row['n_train']:,}"] + [f"{row['scores'][m]:.4f}" for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        f"## Đóng góp của dữ liệu giả lập (`combined` − `{baseline}`)",
        "",
        "| Chỉ số | Chênh lệch |",
        "|---|---|",
    ]
    for metric, delta in comparison["synthetic_contribution"].items():
        lines.append(f"| {metric} | {delta:+.4f} |")

    if comparison["passed"]:
        verdict = (
            f"✅ **ĐẠT** — `combined` vượt `{baseline}` trên cả "
            f"{len(metrics)} chỉ số, đúng mục tiêu thuyết minh "
            '("đạt độ chính xác cao hơn so với phương án nền").'
        )
    else:
        failed = ", ".join(comparison["failed_metrics"]) or "thiếu kết quả `combined`"
        verdict = f"❌ **CHƯA ĐẠT** — `combined` không vượt `{baseline}` ở: {failed}."

    lines += [
        "",
        "## Kết luận",
        "",
        verdict,
        "",
        "## Giới hạn phương pháp",
        "",
        "- Cả 3 kịch bản early-stop và chọn best checkpoint trên tập validation của",
        "  UIT-VSMEC. Kịch bản `synthetic_only` vì vậy có tiếp xúc gián tiếp với dữ liệu",
        "  thật ở khâu chọn checkpoint, dù không mẫu thật nào đi vào gradient. Đây là",
        "  đánh đổi có chủ đích: giữ tiêu chí dừng giống nhau để tập train là biến duy nhất.",
        "- `num_epochs` giống nhau ở cả 3 kịch bản, nên `combined` (nhiều dữ liệu gấp đôi)",
        "  đi qua số bước cập nhật gấp đôi `real_only`.",
        "- Mỗi kịch bản chạy một seed duy nhất; chưa ước lượng phương sai giữa các lần chạy.",
        "",
    ]

    return "\n".join(lines)
