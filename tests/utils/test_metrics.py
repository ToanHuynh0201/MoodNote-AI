"""Tests for evaluation metrics helpers."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.utils.metrics import compute_metrics, get_classification_report, plot_confusion_matrix

LABELS = np.array([0, 0, 1, 1])
PREDICTIONS = np.array([0, 1, 1, 1])  # class indices, not logits
LABEL_NAMES = {0: "Enjoyment", 1: "Sadness"}


def test_compute_metrics_matches_sklearn_reference():
    metrics = compute_metrics(PREDICTIONS, LABELS)

    assert metrics["accuracy"] == accuracy_score(LABELS, PREDICTIONS)
    assert metrics["f1_macro"] == f1_score(LABELS, PREDICTIONS, average="macro")
    assert metrics["f1_weighted"] == f1_score(LABELS, PREDICTIONS, average="weighted")
    assert len(metrics["per_class"]["precision"]) == 2


def test_compute_metrics_accepts_logits():
    logits = np.array([[10.0, 0.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    metrics = compute_metrics(logits, LABELS)
    assert metrics["accuracy"] == accuracy_score(LABELS, PREDICTIONS)


def test_get_classification_report_lists_all_labels():
    report = get_classification_report(PREDICTIONS, LABELS, emotion_labels=LABEL_NAMES)
    assert "Enjoyment" in report
    assert "Sadness" in report


def test_plot_confusion_matrix_returns_figure():
    fig = plot_confusion_matrix(PREDICTIONS, LABELS, emotion_labels=LABEL_NAMES)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
