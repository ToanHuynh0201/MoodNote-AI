"""Tests cho việc ráp 3 tập train ablation (network-free: segmenter được tiêm vào)."""

import pandas as pd
import pytest

from src.data.ablation.build_datasets import (
    build_ablation_datasets,
    load_synthetic_accepted,
    segment_frame,
)


class _FakeSegmenter:
    """Bộ tách từ giả: nối các từ bằng '_' để nhận ra được kết quả đã tách."""

    def segment_text(self, text):
        return "_".join(str(text).split())


def _write_synthetic(dir_path, counts):
    dir_path.mkdir(parents=True, exist_ok=True)
    for split, n in counts.items():
        pd.DataFrame(
            {
                "sample_id": [f"{split}-{i}" for i in range(n)],
                "text": [f"hôm nay {split} {i}" for i in range(n)],
                "label": [i % 7 for i in range(n)],
                "model": ["Qwen3-8B"] * n,
                "generation_round": [1] * n,
            }
        ).to_csv(dir_path / f"{split}.csv", index=False, encoding="utf-8")


def _write_real(dir_path, n):
    dir_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"text": [f"câu_thật số_{i}" for i in range(n)], "label": [i % 7 for i in range(n)]}
    ).to_csv(dir_path / "train.csv", index=False, encoding="utf-8")


def test_load_synthetic_accepted_merges_all_three_splits(tmp_path):
    _write_synthetic(tmp_path, {"train": 5, "validation": 2, "test": 3})

    pool = load_synthetic_accepted(tmp_path)

    assert len(pool) == 10
    assert list(pool.columns) == ["text", "label"]


def test_load_synthetic_accepted_raises_when_a_split_is_missing(tmp_path):
    _write_synthetic(tmp_path, {"train": 5, "validation": 2})

    with pytest.raises(FileNotFoundError):
        load_synthetic_accepted(tmp_path)


def test_segment_frame_segments_text_and_drops_empty_rows():
    df = pd.DataFrame({"text": ["hôm nay vui", "   ", "trời mưa"], "label": [0, 1, 2]})

    out = segment_frame(df, _FakeSegmenter())

    assert out["text"].tolist() == ["hôm_nay_vui", "trời_mưa"]
    assert out["label"].tolist() == [0, 2]


def test_build_ablation_datasets_writes_three_train_sets_with_expected_sizes(tmp_path):
    real_dir = tmp_path / "real"
    accepted_dir = tmp_path / "accepted"
    out_dir = tmp_path / "ablation"
    _write_real(real_dir, 6)
    _write_synthetic(accepted_dir, {"train": 4, "validation": 2, "test": 2})

    counts = build_ablation_datasets(
        real_dir=str(real_dir),
        accepted_dir=str(accepted_dir),
        output_dir=str(out_dir),
        preprocessor=_FakeSegmenter(),
    )

    assert counts == {"real_only": 6, "synthetic_only": 8, "combined": 14}
    for name, expected in counts.items():
        written = pd.read_csv(out_dir / name / "train.csv")
        assert len(written) == expected
        assert list(written.columns) == ["text", "label"]


def test_build_ablation_datasets_segments_synthetic_but_leaves_real_untouched(tmp_path):
    real_dir = tmp_path / "real"
    accepted_dir = tmp_path / "accepted"
    out_dir = tmp_path / "ablation"
    _write_real(real_dir, 1)
    _write_synthetic(accepted_dir, {"train": 1, "validation": 1, "test": 1})

    build_ablation_datasets(
        real_dir=str(real_dir),
        accepted_dir=str(accepted_dir),
        output_dir=str(out_dir),
        preprocessor=_FakeSegmenter(),
    )

    # Dữ liệu thật đã tách từ ở phase 2 — không được tách lại lần nữa.
    assert pd.read_csv(out_dir / "real_only" / "train.csv")["text"].tolist() == ["câu_thật số_0"]
    assert pd.read_csv(out_dir / "synthetic_only" / "train.csv")["text"].tolist() == [
        "hôm_nay_train_0",
        "hôm_nay_validation_0",
        "hôm_nay_test_0",
    ]
