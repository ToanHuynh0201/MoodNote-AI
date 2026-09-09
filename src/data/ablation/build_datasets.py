"""
Ráp 3 tập train cho thí nghiệm ablation (Nội dung 2 của thuyết minh).

Chỉ tập TRAIN thay đổi giữa 3 kịch bản. Validation và test luôn là split chính thức
của UIT-VSMEC (`data/real/processed/{validation,test}.csv`) cho cả 3 — biến duy nhất
của thí nghiệm là dữ liệu huấn luyện.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...utils.config import load_config
from ...utils.logger import get_logger
from ..real.preprocess import VietnamesePreprocessor

logger = get_logger("build_ablation_datasets")

SYNTHETIC_SPLITS = ("train", "validation", "test")


def load_synthetic_accepted(accepted_dir: str | Path) -> pd.DataFrame:
    """
    Gộp cả 3 split synthetic đã qua acceptance gate thành một pool train duy nhất.

    Split nội bộ của synthetic không còn tác dụng vì validation/test của ablation đều
    lấy từ dữ liệu thật — gộp lại để lượng dữ liệu synthetic xấp xỉ lượng dữ liệu thật.

    Args:
        accepted_dir: Thư mục chứa {train,validation,test}.csv đã qua acceptance gate

    Returns:
        DataFrame 2 cột text,label (text CHƯA tách từ)
    """
    accepted_path = Path(accepted_dir)
    frames = []
    for split in SYNTHETIC_SPLITS:
        csv_path = accepted_path / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Không tìm thấy {csv_path} — chạy acceptance gate trước.")
        df = pd.read_csv(csv_path)
        logger.info(f"Đọc {len(df)} mẫu synthetic từ {csv_path}")
        frames.append(df[["text", "label"]])

    pool = pd.concat(frames, ignore_index=True)
    logger.info(f"Tổng pool synthetic: {len(pool)} mẫu")
    return pool


def segment_frame(df: pd.DataFrame, preprocessor: VietnamesePreprocessor) -> pd.DataFrame:
    """
    Tách từ tiếng Việt cho cột text và loại các dòng rỗng sau khi tách.

    Args:
        df: DataFrame có cột text,label
        preprocessor: Bộ tách từ (dùng lại VietnamesePreprocessor của pipeline dữ liệu thật)

    Returns:
        DataFrame mới đã tách từ, index đã reset
    """
    segmented = df.assign(text=[preprocessor.segment_text(t) for t in df["text"]])
    kept = segmented[segmented["text"].str.strip() != ""].reset_index(drop=True)

    n_dropped = len(segmented) - len(kept)
    if n_dropped:
        logger.warning(f"Bỏ {n_dropped} dòng rỗng sau khi tách từ")

    return kept


def build_ablation_datasets(
    real_dir: str = "data/real/processed",
    accepted_dir: str = "data/synthetic/accepted",
    output_dir: str = "data/ablation",
    config_path: str = "configs/model_config.yaml",
    preprocessor: VietnamesePreprocessor | None = None,
) -> dict[str, int]:
    """
    Ghi data/ablation/{real_only,synthetic_only,combined}/train.csv

    Args:
        real_dir: Thư mục dữ liệu thật đã tiền xử lý (đã tách từ sẵn ở phase 2)
        accepted_dir: Thư mục dữ liệu synthetic đã qua acceptance gate
        output_dir: Thư mục gốc để ghi 3 tập train
        config_path: Đường dẫn model_config.yaml (lấy segmenter)
        preprocessor: Bộ tách từ dựng sẵn (tùy chọn — dùng cho test)

    Returns:
        dict: số dòng của từng kịch bản
    """
    if preprocessor is None:
        config = load_config(config_path)
        preprocessor = VietnamesePreprocessor(segmenter=config["preprocessing"]["segmenter"])

    real_train = pd.read_csv(Path(real_dir) / "train.csv")[["text", "label"]]
    logger.info(f"Dữ liệu thật (đã tách từ ở phase 2): {len(real_train)} mẫu")

    logger.info("Tách từ dữ liệu synthetic...")
    synthetic_train = segment_frame(load_synthetic_accepted(accepted_dir), preprocessor)

    scenarios = {
        "real_only": real_train,
        "synthetic_only": synthetic_train,
        "combined": pd.concat([real_train, synthetic_train], ignore_index=True),
    }

    counts = {}
    for name, df in scenarios.items():
        out_path = Path(output_dir) / name
        out_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path / "train.csv", index=False, encoding="utf-8")
        counts[name] = len(df)
        logger.info(f"{name}: {len(df)} mẫu -> {out_path / 'train.csv'}")

    return counts
