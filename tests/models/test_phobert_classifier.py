"""Tests cho FocalLoss và LLRD (network-free: không tải checkpoint PhoBERT nào).

Bỏ qua toàn bộ khi chưa cài torch — CI tối giản chưa cài GPU stack tới phase 6.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from src.models.phobert_classifier import FocalLoss  # noqa: E402


def _logits_and_targets():
    torch.manual_seed(0)
    return torch.randn(8, 7), torch.randint(0, 7, (8,))


def test_focal_loss_with_gamma_zero_equals_cross_entropy():
    logits, targets = _logits_and_targets()

    loss = FocalLoss(gamma=0.0)(logits, targets)

    assert loss.item() == pytest.approx(F.cross_entropy(logits, targets).item(), abs=1e-6)


def test_focal_loss_downweights_easy_examples_relative_to_cross_entropy():
    # Dự đoán rất tự tin và đúng => easy example => focal loss phải nhỏ hơn hẳn CE.
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    targets = torch.tensor([0])

    focal = FocalLoss(gamma=2.0)(logits, targets)

    assert focal.item() < F.cross_entropy(logits, targets).item()


def test_focal_loss_applies_class_weights_when_label_smoothing_is_on():
    logits, targets = _logits_and_targets()
    heavy = torch.ones(7)
    heavy[targets[0]] = 10.0

    unweighted = FocalLoss(gamma=2.0, label_smoothing=0.1)(logits, targets)
    weighted = FocalLoss(gamma=2.0, weight=heavy, label_smoothing=0.1)(logits, targets)

    assert weighted.item() > unweighted.item()


def test_focal_loss_is_finite_with_label_smoothing():
    logits, targets = _logits_and_targets()

    loss = FocalLoss(gamma=2.0, label_smoothing=0.1)(logits, targets)

    assert torch.isfinite(loss)
