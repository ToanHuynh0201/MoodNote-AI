"""
Evaluation metrics for emotion classification
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


EMOTION_LABELS = {
    0: "Enjoyment",
    1: "Sadness",
    2: "Anger",
    3: "Fear",
    4: "Disgust",
    5: "Surprise",
    6: "Other"
}


def compute_metrics(predictions, labels):
    """
    Compute evaluation metrics

    Args:
        predictions: Model predictions (logits or class indices)
        labels: True labels

    Returns:
        dict: Dictionary containing metrics
    """
    # Convert predictions to class indices if needed
    if len(predictions.shape) > 1:
        preds = np.argmax(predictions, axis=1)
    else:
        preds = predictions

    # Overall metrics
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    # Create results dictionary
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        }
    }

    return metrics


def print_metrics(metrics, emotion_labels=EMOTION_LABELS):
    """
    Print metrics in a readable format

    Args:
        metrics: Metrics dictionary from compute_metrics
        emotion_labels: Dictionary mapping label indices to names
    """
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1-Macro:    {metrics['f1_macro']:.4f}")
    print(f"  F1-Weighted: {metrics['f1_weighted']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"{'Emotion':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)

    for i, (prec, rec, f1, sup) in enumerate(zip(
        metrics['per_class']['precision'],
        metrics['per_class']['recall'],
        metrics['per_class']['f1'],
        metrics['per_class']['support']
    )):
        emotion = emotion_labels.get(i, f"Class_{i}")
        print(f"{emotion:<15} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {int(sup):<10}")

    print("=" * 60 + "\n")


def get_classification_report(predictions, labels, emotion_labels=EMOTION_LABELS):
    """
    Get detailed classification report

    Args:
        predictions: Model predictions
        labels: True labels
        emotion_labels: Dictionary mapping label indices to names

    Returns:
        str: Classification report
    """
    # Convert predictions to class indices if needed
    if len(predictions.shape) > 1:
        preds = np.argmax(predictions, axis=1)
    else:
        preds = predictions

    # Get label names in order
    label_names = [emotion_labels[i] for i in sorted(emotion_labels.keys())]

    # Generate report
    report = classification_report(
        labels,
        preds,
        target_names=label_names,
        digits=4
    )

    return report


def plot_confusion_matrix(
    predictions,
    labels,
    emotion_labels=EMOTION_LABELS,
    save_path=None,
    figsize=(10, 8)
):
    """
    Plot confusion matrix

    Args:
        predictions: Model predictions
        labels: True labels
        emotion_labels: Dictionary mapping label indices to names
        save_path: Path to save the plot (optional)
        figsize: Figure size

    Returns:
        matplotlib.figure.Figure: The confusion matrix figure
    """
    # Convert predictions to class indices if needed
    if len(predictions.shape) > 1:
        preds = np.argmax(predictions, axis=1)
    else:
        preds = predictions

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)

    # Get label names in order
    label_names = [emotion_labels[i] for i in sorted(emotion_labels.keys())]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    return fig


def compute_metrics_for_trainer(eval_pred):
    """
    Compute metrics for Hugging Face Trainer

    Args:
        eval_pred: EvalPrediction object with predictions and label_ids

    Returns:
        dict: Metrics dictionary
    """
    predictions, labels = eval_pred
    metrics = compute_metrics(predictions, labels)

    # Return only scalar metrics for Trainer
    return {
        'accuracy': metrics['accuracy'],
        'f1_macro': metrics['f1_macro'],
        'f1_weighted': metrics['f1_weighted']
    }


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics computation...")

    # Create dummy predictions and labels
    np.random.seed(42)
    n_samples = 100
    n_classes = 7

    # Simulate predictions (logits)
    predictions = np.random.randn(n_samples, n_classes)

    # Simulate true labels
    labels = np.random.randint(0, n_classes, n_samples)

    # Compute metrics
    metrics = compute_metrics(predictions, labels)

    # Print metrics
    print_metrics(metrics)

    # Print classification report
    print("\nClassification Report:")
    print(get_classification_report(predictions, labels))

    # Plot confusion matrix
    fig = plot_confusion_matrix(predictions, labels)
    plt.show()
