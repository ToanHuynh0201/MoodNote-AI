"""
Vietnamese text preprocessing with word segmentation for PhoBERT
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ...utils.config import load_config
from ...utils.logger import get_logger

logger = get_logger("preprocess")

try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except ImportError:
    ViTokenizer = None  # type: ignore[assignment]
    PYVI_AVAILABLE = False
    logger.warning("pyvi not installed. Install with: pip install pyvi")

_LABEL_COLUMN_NAMES = ("label", "labels", "emotion")
_TEXT_COLUMN_PRIORITY = ("sentence", "text", "content")


class VietnamesePreprocessor:
    """Vietnamese text preprocessor with word segmentation"""

    def __init__(self, segmenter: str = "pyvi") -> None:
        """
        Initialize preprocessor

        Args:
            segmenter: Type of segmenter to use ("pyvi" or "vncorenlp")
        """
        self.segmenter = segmenter

        if segmenter == "pyvi" and not PYVI_AVAILABLE:
            raise ImportError("pyvi is not installed. Install with: pip install pyvi")

    def segment_text(self, text: str) -> str:
        """
        Segment Vietnamese text into words

        Args:
            text: Input Vietnamese text

        Returns:
            Word-segmented text (e.g., "hôm nay" -> "hôm_nay")
        """
        if not isinstance(text, str):
            return ""

        text = text.strip()
        if not text:
            return ""

        if self.segmenter == "pyvi":
            # Use pyvi for word segmentation
            assert ViTokenizer is not None
            segmented = ViTokenizer.tokenize(text)
            return segmented
        else:
            raise ValueError(f"Unsupported segmenter: {self.segmenter}")

    def preprocess_text(self, text: str, lowercase: bool = False) -> str:
        """
        Preprocess Vietnamese text

        Args:
            text: Input text
            lowercase: Whether to lowercase the text

        Returns:
            Preprocessed text
        """
        # Segment text
        text = self.segment_text(text)

        # Optional lowercase
        if lowercase:
            text = text.lower()

        return text


def _detect_text_column(columns: list[str]) -> tuple[str, bool]:
    """
    Detect the text column, case-insensitively.

    Returns:
        (column_name, used_fallback) — used_fallback is True when no column
        matched _TEXT_COLUMN_PRIORITY and the "first non-label column" guess
        was used instead.
    """
    lowered = {col.lower(): col for col in columns}
    for candidate in _TEXT_COLUMN_PRIORITY:
        if candidate in lowered:
            return lowered[candidate], False

    fallback = next(col for col in columns if col.lower() not in _LABEL_COLUMN_NAMES)
    return fallback, True


def _detect_label_column(columns: list[str]) -> str:
    """Detect the label column, case-insensitively (defaults to the last column)."""
    for col in columns:
        if col.lower() in _LABEL_COLUMN_NAMES:
            return col
    return columns[-1]


def find_exact_text_leakage(
    train_texts: list[str], validation_texts: list[str], test_texts: list[str]
) -> set[str]:
    """
    Find exact-text overlap (normalized: stripped + lowercased) between the
    test split and train/validation.

    This is a read-only check: callers must not use the result to drop or
    modify rows — the test split stays fixed exactly as published.

    Args:
        train_texts: Raw text values from the train split
        validation_texts: Raw text values from the validation split
        test_texts: Raw text values from the test split

    Returns:
        Set of normalized texts that appear in both test and train/validation
    """

    def _norm(text: object) -> str:
        return str(text).strip().lower()

    train_val = {_norm(t) for t in train_texts} | {_norm(t) for t in validation_texts}
    return {_norm(t) for t in test_texts} & train_val


def _log_leakage_guard(raw: dict[str, pd.DataFrame], text_col: str) -> None:
    """Run the leakage guard across whatever splits are available and log the result."""
    if not {'train', 'validation', 'test'} <= raw.keys():
        logger.info("Leakage guard skipped: not all three splits present.")
        return

    leaked = find_exact_text_leakage(
        raw['train'][text_col].tolist(),
        raw['validation'][text_col].tolist(),
        raw['test'][text_col].tolist(),
    )
    if leaked:
        logger.warning(
            f"Leakage guard: {len(leaked)} test examples also appear in "
            f"train/validation (normalized exact-text match). Test split left "
            f"unmodified — VSMEC official split is fixed. "
            f"Examples: {sorted(leaked)[:5]}"
        )
    else:
        logger.info("Leakage guard: no exact-text overlap between test and train/validation.")


def preprocess_dataset(
    input_dir: str = "data/real/raw",
    output_dir: str = "data/real/processed",
    config_path: str = "configs/model_config.yaml",
) -> None:
    """
    Preprocess UIT-VSMEC dataset with Vietnamese word segmentation

    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory to save preprocessed files
        config_path: Path to model configuration file
    """
    logger.info("Starting Vietnamese text preprocessing...")

    # Load configuration
    config = load_config(config_path)

    segmenter = config['preprocessing']['segmenter']
    lowercase = config['preprocessing'].get('lowercase', False)

    # Initialize preprocessor
    preprocessor = VietnamesePreprocessor(segmenter=segmenter)
    logger.info(f"Using segmenter: {segmenter}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load whichever splits are present up front, so the leakage guard can
    # compare across splits before any file is written.
    raw: dict[str, pd.DataFrame] = {}
    for split_name in ['train', 'validation', 'test']:
        input_file = Path(input_dir) / f"{split_name}.csv"

        if not input_file.exists():
            logger.warning(f"{input_file} not found. Skipping...")
            continue

        raw[split_name] = pd.read_csv(input_file)
        logger.info(f"Loaded {len(raw[split_name])} samples from {split_name} split")

    if not raw:
        logger.warning("No raw splits found, nothing to preprocess.")
        return

    # Detect columns once — the raw VSMEC splits share an identical schema.
    columns = list(next(iter(raw.values())).columns)
    text_col, used_fallback = _detect_text_column(columns)
    if used_fallback:
        logger.warning(f"Text column not recognized by name, guessing '{text_col}'.")
    logger.info(f"Using column '{text_col}' for text")

    label_col = _detect_label_column(columns)
    logger.info(f"Using column '{label_col}' for labels")

    _log_leakage_guard(raw, text_col)

    # Build label mapping from config (invert: "Enjoyment" -> 0)
    emotion_labels = config['emotion_labels']
    label_to_int = {v: int(k) for k, v in emotion_labels.items()}

    # Process each available split
    for split_name, df in raw.items():
        logger.info(f"Processing {split_name} split...")

        # Preprocess texts
        logger.info("Applying word segmentation...")
        segmented_texts = []

        for text in tqdm(df[text_col], desc=f"Segmenting {split_name}"):
            segmented = preprocessor.preprocess_text(text, lowercase=lowercase)
            segmented_texts.append(segmented)

        # Convert string labels to int
        raw_labels = df[label_col].tolist()
        if label_to_int and isinstance(raw_labels[0], str):
            int_labels = [label_to_int[lbl] for lbl in raw_labels]
        else:
            int_labels = [int(lbl) for lbl in raw_labels]

        # Create new DataFrame
        processed_df = pd.DataFrame({
            'text': segmented_texts,
            'label': int_labels
        })

        # Save preprocessed data
        output_file = output_path / f"{split_name}.csv"
        processed_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Saved preprocessed data to {output_file}")

        # Show examples
        for i in range(min(3, len(df))):
            logger.debug(f"Original:  {df[text_col].iloc[i]}")
            logger.debug(f"Segmented: {processed_df['text'].iloc[i]}")
            logger.debug(f"Label:     {processed_df['label'].iloc[i]}")

    logger.info("Preprocessing complete!")


def main() -> None:
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess Vietnamese emotion dataset")
    parser.add_argument(
        "--input-dir",
        default="data/real/raw",
        help="Directory containing raw CSV files (default: data/real/raw)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/real/processed",
        help="Directory to save preprocessed files (default: data/real/processed)"
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to model config YAML (default: configs/model_config.yaml)"
    )
    args = parser.parse_args()
    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
