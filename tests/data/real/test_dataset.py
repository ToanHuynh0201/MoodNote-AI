"""Tests for the EmotionDataset PyTorch wrapper (network-free: tokenizer is injected/mocked).

Skipped entirely when torch/transformers aren't installed (CI's minimal
dependency set intentionally excludes them until phase 6).
"""

import inspect

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import src.data.real.dataset as dataset_module  # noqa: E402
from src.data.real.dataset import EmotionDataset, create_dataloaders  # noqa: E402


class _FakeTokenizer:
    """Stands in for a HuggingFace tokenizer — same call interface, no network/model."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts, max_length, padding, truncation, return_tensors):
        self.calls.append(list(texts))
        n = len(texts)
        return {
            "input_ids": torch.ones((n, max_length), dtype=torch.long),
            "attention_mask": torch.ones((n, max_length), dtype=torch.long),
        }


def _write_csv(path, texts, labels):
    pd.DataFrame({"text": texts, "label": labels}).to_csv(path, index=False, encoding="utf-8")


def test_emotion_dataset_length_matches_csv_rows(tmp_path):
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, ["vui qua", "buon qua", "gian qua"], [0, 1, 2])

    ds = EmotionDataset(str(csv_path), tokenizer=_FakeTokenizer(), max_length=8)

    assert len(ds) == 3


def test_emotion_dataset_getitem_returns_expected_keys_and_shapes(tmp_path):
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, ["vui qua"], [0])

    ds = EmotionDataset(str(csv_path), tokenizer=_FakeTokenizer(), max_length=8)
    item = ds[0]

    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].shape == (8,)


def test_emotion_dataset_labels_dtype_is_long(tmp_path):
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, ["vui qua"], [3])

    ds = EmotionDataset(str(csv_path), tokenizer=_FakeTokenizer(), max_length=8)

    assert ds[0]["labels"].dtype == torch.long
    assert ds[0]["labels"].item() == 3


def test_emotion_dataset_uses_injected_tokenizer_without_network_call(tmp_path, monkeypatch):
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, ["vui qua"], [0])

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("AutoTokenizer.from_pretrained should not be called")

    monkeypatch.setattr(dataset_module.AutoTokenizer, "from_pretrained", _raise_if_called)

    EmotionDataset(str(csv_path), tokenizer=_FakeTokenizer(), max_length=8)


def test_emotion_dataset_pretokenizes_all_texts_in_one_call(tmp_path):
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, ["vui qua", "buon qua"], [0, 1])
    fake_tokenizer = _FakeTokenizer()

    EmotionDataset(str(csv_path), tokenizer=fake_tokenizer, max_length=8)

    assert len(fake_tokenizer.calls) == 1
    assert fake_tokenizer.calls[0] == ["vui qua", "buon qua"]


def test_emotion_dataset_default_tokenizer_name_matches_configured_model():
    default = inspect.signature(EmotionDataset.__init__).parameters["tokenizer_name"].default
    assert default == "vinai/phobert-base-v2"


def test_create_dataloaders_builds_three_loaders_sharing_one_tokenizer(tmp_path, monkeypatch):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "validation.csv"
    test_csv = tmp_path / "test.csv"
    _write_csv(train_csv, ["vui qua", "buon qua"], [0, 1])
    _write_csv(val_csv, ["vui"], [0])
    _write_csv(test_csv, ["buon"], [1])

    call_count = {"n": 0}

    def _fake_from_pretrained(name):
        call_count["n"] += 1
        return _FakeTokenizer()

    monkeypatch.setattr(dataset_module.AutoTokenizer, "from_pretrained", _fake_from_pretrained)

    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        str(train_csv), str(val_csv), str(test_csv), batch_size=2, max_length=8
    )

    assert call_count["n"] == 1
    batch = next(iter(train_loader))
    assert batch["input_ids"].shape[1] == 8
