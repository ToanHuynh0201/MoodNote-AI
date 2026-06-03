"""
Back-translation augmentation — chạy trên Colab trước khi train.

Thay thế train_augmented.csv (đã được tạo local bằng swap/insertion) bằng phiên bản
có back-translation cho Enjoyment, Anger, Surprise.

Cách dùng (trên Colab):
    !pip install deep_translator -q
    !python /content/MoodNote-AI/scripts/augment_colab.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("augment_colab")

TRAIN_CSV = REPO_ROOT / "data" / "processed" / "train.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "train_augmented.csv"
TEST_CSV = REPO_ROOT / "data" / "processed" / "test.csv"


def main() -> None:
    try:
        from deep_translator import GoogleTranslator  # noqa: F401
    except ImportError:
        logger.error("deep_translator not found. Run: pip install deep_translator")
        sys.exit(1)

    import pandas as pd

    from src.data.augment import (
        DEFAULT_AUGMENT_TARGETS,
        DEFAULT_CLASS_TECHNIQUES,
        augment_dataset,
    )

    logger.info("Augmentation with back-translation (Colab)")
    logger.info("Classes using back_translation: Enjoyment(0), Anger(2), Surprise(5)")
    logger.info("Classes using swap/insertion  : Fear(3), Disgust(4)")

    augment_dataset(
        input_csv=str(TRAIN_CSV),
        output_csv=str(OUTPUT_CSV),
        target_counts=DEFAULT_AUGMENT_TARGETS,
        techniques=["swap", "insertion"],
        class_techniques=DEFAULT_CLASS_TECHNIQUES,
        seed=42,
    )

    # Leakage prevention
    if TEST_CSV.exists():
        test_texts = set(pd.read_csv(TEST_CSV)["text"].str.strip().str.lower())
        aug_df = pd.read_csv(OUTPUT_CSV)
        before = len(aug_df)
        aug_df = aug_df[~aug_df["text"].str.strip().str.lower().isin(test_texts)]
        n_removed = before - len(aug_df)
        if n_removed:
            aug_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
            logger.info(f"Removed {n_removed} augmented samples overlapping with test set.")

    logger.info("Done! train_augmented.csv ready for training.")


if __name__ == "__main__":
    main()
