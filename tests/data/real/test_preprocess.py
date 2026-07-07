"""Tests for the VSMEC preprocessing pipeline (word segmentation + leakage guard)."""

import pandas as pd
import pytest

from src.data.real.preprocess import (
    VietnamesePreprocessor,
    _detect_label_column,
    _detect_text_column,
    find_exact_text_leakage,
    preprocess_dataset,
)

_CONFIG_YAML = """
emotion_labels:
    0: "Enjoyment"
    1: "Sadness"
preprocessing:
    segmenter: "pyvi"
    lowercase: false
"""


def _write_config(tmp_path, content=_CONFIG_YAML):
    config_file = tmp_path / "model_config.yaml"
    config_file.write_text(content, encoding="utf-8")
    return config_file


def _write_raw_csv(path, texts, labels):
    pd.DataFrame({"Sentence": texts, "Emotion": labels}).to_csv(
        path, index=False, encoding="utf-8"
    )


def test_detect_text_column_is_case_insensitive_and_prefers_sentence():
    col, used_fallback = _detect_text_column(["Sentence", "Emotion"])
    assert col == "Sentence"
    assert used_fallback is False


def test_detect_text_column_falls_back_when_no_known_name():
    col, used_fallback = _detect_text_column(["weird_text_col", "Emotion"])
    assert col == "weird_text_col"
    assert used_fallback is True


def test_detect_label_column_is_case_insensitive():
    assert _detect_label_column(["Sentence", "EMOTION"]) == "EMOTION"


def test_segment_text_empty_and_none_return_empty_string():
    preprocessor = VietnamesePreprocessor(segmenter="pyvi")
    assert preprocessor.segment_text("") == ""
    assert preprocessor.segment_text(None) == ""
    assert preprocessor.segment_text("   ") == ""


def test_preprocess_text_respects_lowercase_flag():
    preprocessor = VietnamesePreprocessor(segmenter="pyvi")
    segmented = preprocessor.preprocess_text("Hôm Nay", lowercase=False)
    lowered = preprocessor.preprocess_text("Hôm Nay", lowercase=True)
    assert lowered == segmented.lower()


def test_find_exact_text_leakage_detects_manufactured_overlap():
    leaked = find_exact_text_leakage(
        train_texts=["a", "b"],
        validation_texts=["c"],
        test_texts=["B", " a  ", "d"],
    )
    assert leaked == {"a", "b"}


def test_find_exact_text_leakage_no_overlap_returns_empty_set():
    leaked = find_exact_text_leakage(
        train_texts=["a", "b"], validation_texts=["c"], test_texts=["d", "e"]
    )
    assert leaked == set()


def test_preprocess_dataset_writes_text_and_label_columns_with_correct_int_mapping(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_raw_csv(raw_dir / "train.csv", ["vui qua", "buon qua"], ["Enjoyment", "Sadness"])
    _write_raw_csv(raw_dir / "validation.csv", ["vui"], ["Enjoyment"])
    _write_raw_csv(raw_dir / "test.csv", ["buon"], ["Sadness"])
    config_file = _write_config(tmp_path)

    preprocess_dataset(
        input_dir=str(raw_dir), output_dir=str(out_dir), config_path=str(config_file)
    )

    train = pd.read_csv(out_dir / "train.csv")
    assert list(train.columns) == ["text", "label"]
    assert train["label"].tolist() == [0, 1]


def test_preprocess_dataset_matches_real_model_config_label_mapping(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_raw_csv(raw_dir / "train.csv", ["vui qua", "buon qua"], ["Enjoyment", "Sadness"])

    preprocess_dataset(
        input_dir=str(raw_dir),
        output_dir=str(out_dir),
        config_path="configs/model_config.yaml",
    )

    train = pd.read_csv(out_dir / "train.csv")
    assert train["label"].tolist() == [0, 1]


def test_preprocess_dataset_skips_missing_split_with_warning(tmp_path, caplog):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_raw_csv(raw_dir / "train.csv", ["vui qua"], ["Enjoyment"])
    config_file = _write_config(tmp_path)

    with caplog.at_level("WARNING"):
        preprocess_dataset(
            input_dir=str(raw_dir), output_dir=str(out_dir), config_path=str(config_file)
        )

    assert (out_dir / "train.csv").exists()
    assert not (out_dir / "validation.csv").exists()
    assert not (out_dir / "test.csv").exists()
    assert "not found" in caplog.text


def test_preprocess_dataset_missing_emotion_labels_raises_keyerror(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw_csv(raw_dir / "train.csv", ["vui qua"], ["Enjoyment"])
    bad_config = _write_config(
        tmp_path, content='preprocessing:\n    segmenter: "pyvi"\n    lowercase: false\n'
    )

    with pytest.raises(KeyError):
        preprocess_dataset(
            input_dir=str(raw_dir),
            output_dir=str(tmp_path / "processed"),
            config_path=str(bad_config),
        )


def test_preprocess_dataset_logs_leakage_warning_and_leaves_test_rows_untouched(tmp_path, caplog):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_raw_csv(raw_dir / "train.csv", ["Vui Qua"], ["Enjoyment"])
    _write_raw_csv(raw_dir / "validation.csv", ["khong lien quan"], ["Sadness"])
    _write_raw_csv(raw_dir / "test.csv", ["vui qua"], ["Enjoyment"])
    config_file = _write_config(tmp_path)

    with caplog.at_level("WARNING"):
        preprocess_dataset(
            input_dir=str(raw_dir), output_dir=str(out_dir), config_path=str(config_file)
        )

    assert "Leakage guard" in caplog.text
    test_out = pd.read_csv(out_dir / "test.csv")
    assert len(test_out) == 1
