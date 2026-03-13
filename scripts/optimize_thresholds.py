"""
Per-class threshold optimization for multiclass emotion classification.

Instead of argmax(probs), predict class i = argmax(probs / thresholds).
This allows biasing the model toward or away from specific classes.

Thresholds are optimized on the validation set to maximize F1-Macro,
then evaluated on the test set.

Usage (local):
    python scripts/optimize_thresholds.py

Usage (Colab):
    %run scripts/optimize_thresholds.py \
        --model-dir /content/drive/MyDrive/MoodNote-AI/best_model \
        --data-dir  /content/MoodNote-AI/data/processed \
        --output    /content/drive/MyDrive/MoodNote-AI/thresholds.json
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from src.data.dataset import EmotionDataset
from src.models.model_utils import load_model
from src.utils.metrics import compute_metrics, print_metrics
from src.utils.logger import setup_logger

logger = setup_logger(name="threshold_opt")

EMOTION_LABELS = {
    0: "Enjoyment", 1: "Sadness", 2: "Anger",
    3: "Fear", 4: "Disgust", 5: "Surprise", 6: "Other"
}
NUM_CLASSES = 7


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def get_probabilities(model, dataset, batch_size=64, device="cpu"):
    """Run model inference and return softmax probabilities + true labels."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def predict_with_thresholds(probs, thresholds):
    """
    Per-class threshold prediction.
    Computes scaled_probs[i] = probs[i] / thresholds[i], then argmax.
    Equivalent to adding a per-class log-bias to logits.
    """
    scaled = probs / np.array(thresholds)
    return np.argmax(scaled, axis=1)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def f1_macro_neg(thresholds, probs, labels):
    """Objective: negative F1-Macro (scipy minimizes)."""
    preds = predict_with_thresholds(probs, thresholds)
    metrics = compute_metrics(preds, labels)
    return -metrics["f1_macro"]


def optimize_thresholds(val_probs, val_labels):
    """
    Optimize per-class thresholds on the validation set using
    differential evolution (global optimizer, no gradient needed).
    """
    logger.info("Optimizing thresholds via differential evolution...")
    bounds = [(0.05, 3.0)] * NUM_CLASSES

    result = differential_evolution(
        func=f1_macro_neg,
        bounds=bounds,
        args=(val_probs, val_labels),
        seed=42,
        maxiter=300,
        popsize=15,
        tol=1e-4,
        workers=1,
        polish=True,
    )

    thresholds = result.x.tolist()
    logger.info(f"Optimization converged: {result.success}  |  val F1-Macro: {-result.fun:.4f}")
    return thresholds


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(probs, labels, thresholds, split_name):
    preds = predict_with_thresholds(probs, thresholds)
    metrics = compute_metrics(preds, labels)
    print(f"\n--- {split_name} (optimized thresholds) ---")
    print_metrics(metrics, EMOTION_LABELS)
    return metrics


def evaluate_baseline(probs, labels, split_name):
    preds = np.argmax(probs, axis=1)
    metrics = compute_metrics(preds, labels)
    print(f"\n--- {split_name} (baseline argmax) ---")
    print_metrics(metrics, EMOTION_LABELS)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",  default="models/best_model")
    parser.add_argument("--data-dir",   default="data/processed")
    parser.add_argument("--output",     default="models/best_model/thresholds.json")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model_dir} ...")
    model, tokenizer = load_model(args.model_dir, device=device)
    model_name = model.model_name

    # Load datasets
    data_path = Path(args.data_dir)
    val_dataset  = EmotionDataset(data_path / "validation.csv", model_name, max_length=128)
    test_dataset = EmotionDataset(data_path / "test.csv",       model_name, max_length=128)
    logger.info(f"Val: {len(val_dataset)}  |  Test: {len(test_dataset)}")

    # Get probabilities
    logger.info("Running inference on validation set...")
    val_probs, val_labels = get_probabilities(model, val_dataset, args.batch_size, device)

    logger.info("Running inference on test set...")
    test_probs, test_labels = get_probabilities(model, test_dataset, args.batch_size, device)

    # Baseline
    val_base  = evaluate_baseline(val_probs,  val_labels,  "Validation")
    test_base = evaluate_baseline(test_probs, test_labels, "Test")

    # Optimize on validation
    thresholds = optimize_thresholds(val_probs, val_labels)

    # Print per-class thresholds
    print("\nOptimized thresholds:")
    for i, t in enumerate(thresholds):
        print(f"  {EMOTION_LABELS[i]:<12}: {t:.4f}")

    # Evaluate with optimized thresholds
    val_opt  = evaluate(val_probs,  val_labels,  thresholds, "Validation")
    test_opt = evaluate(test_probs, test_labels, thresholds, "Test")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'':20} {'Baseline':>10} {'Optimized':>10} {'Delta':>8}")
    print("-" * 50)
    for metric in ("accuracy", "f1_macro", "f1_weighted"):
        b = test_base[metric]
        o = test_opt[metric]
        print(f"Test {metric:<15} {b:>10.4f} {o:>10.4f} {o-b:>+8.4f}")

    # Save thresholds
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thresholds": thresholds,
        "per_class": {EMOTION_LABELS[i]: thresholds[i] for i in range(NUM_CLASSES)},
        "val_f1_macro_baseline":  val_base["f1_macro"],
        "val_f1_macro_optimized": val_opt["f1_macro"],
        "test_f1_macro_baseline":  test_base["f1_macro"],
        "test_f1_macro_optimized": test_opt["f1_macro"],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Thresholds saved to {output_path}")


if __name__ == "__main__":
    main()
