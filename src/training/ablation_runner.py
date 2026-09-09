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
import csv
import json
import statistics
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
    seed: int | None = None,
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
        results_dir: Nơi ghi ablation_<scenario>_seed<N>.json và confusion matrix
        use_wandb: Bật/tắt logging W&B
        smoke: Chạy thử rút gọn (32 dòng mỗi split, 1 epoch) để kiểm tra đường ống
        seed: Seed cho lần chạy này; None → lấy training_config["training"]["seed"].
            File kết quả gắn hậu tố _seed<N> nên chạy nhiều seed không ghi đè nhau.

    Returns:
        dict kết quả (đồng thời ghi ra reports/ablation_<scenario>_seed<N>.json)
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Kịch bản không hợp lệ: {scenario!r}. Chọn trong {SCENARIOS}.")

    # Import nặng để trong hàm: compare_scenarios/render_comparison_markdown phải
    # dùng được ở môi trường không cài torch/transformers.
    from transformers import set_seed

    from ..data.real.dataset import EmotionDataset
    from ..models.model_utils import get_device, print_model_summary, save_model
    from ..models.phobert_classifier import create_model
    from ..utils.metrics import compute_metrics, plot_confusion_matrix, print_metrics
    from .trainer import train_model

    eff_seed = seed if seed is not None else int(training_config["training"]["seed"])
    # Phải seed TRƯỚC create_model: head classifier (nn.Linear/nn.LayerNorm) khởi tạo ở
    # đó, nếu không set_seed thì mỗi lần chạy ra head khác nhau ngoài kiểm soát → phương
    # sai giữa các seed không phản ánh đúng. Trainer của HF cũng set_seed nội bộ nhưng
    # SAU khi model đã dựng.
    set_seed(eff_seed)
    sfx = f"_seed{eff_seed}"

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
    run_config["training"]["seed"] = eff_seed  # TrainingArguments(seed=...) khớp eff_seed
    wandb_cfg = run_config.setdefault("wandb", {})
    wandb_cfg["name"] = f"{wandb_cfg.get('name', 'phobert')}-{scenario}{sfx}"

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
        save_path=results_path / f"confusion_matrix_{scenario}{sfx}.png",
    )

    # Dự đoán từng câu -> cho phép kiểm định ghép cặp (McNemar) mà không cần train lại.
    preds_file = results_path / f"preds_{scenario}{sfx}.csv"
    pred_ids = predictions.predictions.argmax(axis=1).tolist()
    true_ids = predictions.label_ids.tolist()
    with preds_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pred", "true"])
        writer.writerows(zip(pred_ids, true_ids, strict=True))
    logger.info(f"[{scenario}] Đã ghi dự đoán từng câu -> {preds_file}")

    model_dir = f"models/best_model/{scenario}{sfx}"
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
        "seed": eff_seed,
        "num_epochs": run_config["training"]["num_epochs"],
        "train_csv": train_path,
        "validation_csv": validation_path,
        "test_csv": test_path,
        "n_train": len(datasets["train"]),
        "n_validation": len(datasets["validation"]),
        "n_test": len(datasets["test"]),
        "model_dir": model_dir,
        "preds_csv": str(preds_file),
        "metrics": metrics,
    }

    out_file = results_path / f"ablation_{scenario}{sfx}.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[{scenario}] Đã ghi kết quả -> {out_file}")

    return result


def aggregate_seeds(results: list[dict]) -> dict[str, Any]:
    """
    Gộp nhiều kết quả run_scenario() CÙNG một kịch bản (khác seed) thành một dict
    "kết quả trung bình" có hình dạng như run_scenario() trả về, để đưa thẳng vào
    compare_scenarios(). Hàm thuần — chỉ statistics stdlib.

    Args:
        results: list dict kết quả (>= 1 phần tử), cùng scenario

    Returns:
        dict gồm scenario/smoke/model_name/n_train, `metrics` = mean từng chỉ số
        overall, `metrics_std` = độ lệch chuẩn quần thể (pstdev), `n_seeds`, `seeds`.
        list 1 phần tử → mọi std = 0.0.
    """
    if not results:
        raise ValueError("aggregate_seeds() nhận list rỗng.")

    metric_keys = [k for k, v in results[0]["metrics"].items() if isinstance(v, (int, float))]
    per_metric = {k: [r["metrics"][k] for r in results] for k in metric_keys}

    first = results[0]
    return {
        "scenario": first["scenario"],
        "smoke": any(r.get("smoke") for r in results),
        "model_name": first.get("model_name"),
        "n_train": first["n_train"],
        "metrics": {k: statistics.fmean(v) for k, v in per_metric.items()},
        "metrics_std": {k: statistics.pstdev(v) for k, v in per_metric.items()},
        "n_seeds": len(results),
        "seeds": [r.get("seed") for r in results],
    }


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
        std = row.get("scores_std") or {}
        score_cells = [
            f"{row['scores'][m]:.4f} ± {std[m]:.4f}"
            if std.get(m) is not None
            else f"{row['scores'][m]:.4f}"
            for m in metrics
        ]
        cells = [tag, f"{row['n_train']:,}", *score_cells]
        lines.append("| " + " | ".join(cells) + " |")

    n_seeds = comparison.get("n_seeds") or {}
    if n_seeds:
        parts = ", ".join(f"`{k}` {v} seed" for k, v in n_seeds.items())
        lines += [
            "",
            f"_Số lần chạy: {parts}. Ô số dạng mean ± độ lệch chuẩn quần thể qua các seed._",
        ]

    lines += [
        "",
        f"## Đóng góp của dữ liệu giả lập (`combined` − `{baseline}`)",
        "",
        "| Chỉ số | Chênh lệch |",
        "|---|---|",
    ]
    for metric, delta in comparison["synthetic_contribution"].items():
        lines.append(f"| {metric} | {delta:+.4f} |")

    mcn = comparison.get("mcnemar")
    if mcn:
        lines += [
            "",
            f"McNemar (`combined` vs `{baseline}`, seed {mcn['seed']}, cùng {mcn['n']} câu "
            f"test): p = {mcn['p_value']:.4g} — chỉ `combined` đúng: {mcn['only_treat_correct']}, "
            f"chỉ `{baseline}` đúng: {mcn['only_base_correct']}.",
        ]

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
        "- `real_only` và `combined` chạy nhiều seed, số báo cáo là mean ± độ lệch chuẩn.",
        "  `synthetic_only` chỉ chạy một seed — là mốc control, không vào tiêu chí pass/fail.",
        "- Cả 3 kịch bản dùng chung `num_epochs` + `early_stopping_patience` để mỗi model",
        "  train tới hội tụ trên tập validation chung; không ép bằng `max_steps`, nên",
        "  `combined` (gấp đôi dữ liệu) vẫn đi nhiều bước optimizer hơn `real_only` trong",
        "  cùng số epoch, và lịch learning-rate cosine giãn theo `num_epochs`.",
        "",
    ]

    return "\n".join(lines)
