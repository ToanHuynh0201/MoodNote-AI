"""Tests for the UIT-VSMEC download helper (network-free: load_dataset is mocked)."""

import pandas as pd

import src.data.real.download_vsmec as download_vsmec


class _FakeSplit:
    """Stands in for a HuggingFace `datasets.Dataset` split."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def __len__(self) -> int:
        return len(self._df)

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()


def _fake_dataset(train_df, val_df=None, test_df=None):
    return {
        "train": _FakeSplit(train_df),
        "validation": _FakeSplit(val_df if val_df is not None else train_df.iloc[:1]),
        "test": _FakeSplit(test_df if test_df is not None else train_df.iloc[:1]),
    }


def _patch_load_dataset(monkeypatch, dataset):
    monkeypatch.setattr(download_vsmec, "load_dataset", lambda name: dataset)


def test_download_uit_vsmec_writes_csv_per_split_with_original_columns(tmp_path, monkeypatch):
    df = pd.DataFrame({"Sentence": ["vui qua", "buon qua"], "Emotion": ["Enjoyment", "Sadness"]})
    _patch_load_dataset(monkeypatch, _fake_dataset(df))

    download_vsmec.download_uit_vsmec(output_dir=str(tmp_path))

    for split in ("train", "validation", "test"):
        out = pd.read_csv(tmp_path / f"{split}.csv")
        assert list(out.columns) == ["Sentence", "Emotion"]


def test_download_uit_vsmec_returns_dataframes_dict_keyed_by_split(tmp_path, monkeypatch):
    df = pd.DataFrame({"Sentence": ["vui qua"], "Emotion": ["Enjoyment"]})
    _patch_load_dataset(monkeypatch, _fake_dataset(df))

    result = download_vsmec.download_uit_vsmec(output_dir=str(tmp_path))

    assert set(result.keys()) == {"train", "validation", "test"}
    assert isinstance(result["train"], pd.DataFrame)


def test_download_uit_vsmec_logs_distribution_for_string_emotion_column(
    tmp_path, monkeypatch, caplog
):
    df = pd.DataFrame(
        {"Sentence": ["a", "b", "c"], "Emotion": ["Enjoyment", "Enjoyment", "Sadness"]}
    )
    _patch_load_dataset(monkeypatch, _fake_dataset(df))

    with caplog.at_level("INFO"):
        download_vsmec.download_uit_vsmec(output_dir=str(tmp_path))

    assert "Enjoyment" in caplog.text
    assert "Sadness" in caplog.text


def test_download_uit_vsmec_falls_back_to_numeric_label_column(tmp_path, monkeypatch, caplog):
    df = pd.DataFrame({"Sentence": ["a", "b"], "label": [0, 1]})
    _patch_load_dataset(monkeypatch, _fake_dataset(df))

    with caplog.at_level("INFO"):
        result = download_vsmec.download_uit_vsmec(output_dir=str(tmp_path))

    assert "Enjoyment" in caplog.text  # DEFAULT_EMOTION_LABELS[0]
    assert "Sadness" in caplog.text  # DEFAULT_EMOTION_LABELS[1]
    assert list(result["train"].columns) == ["Sentence", "label"]


def test_download_uit_vsmec_unknown_column_layout_logs_warning_without_crashing(
    tmp_path, monkeypatch, caplog
):
    df = pd.DataFrame({"Sentence": ["a", "b"], "weird_col": [1, 2]})
    _patch_load_dataset(monkeypatch, _fake_dataset(df))

    with caplog.at_level("WARNING"):
        download_vsmec.download_uit_vsmec(output_dir=str(tmp_path))

    assert "Unknown column layout" in caplog.text
