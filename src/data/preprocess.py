"""
Vietnamese text preprocessing with word segmentation for PhoBERT
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from ..utils.logger import get_logger

logger = get_logger("preprocess")

try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except ImportError:
    ViTokenizer = None  # type: ignore[assignment]
    PYVI_AVAILABLE = False
    logger.warning("pyvi not installed. Install with: pip install pyvi")


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


def preprocess_dataset(
    input_dir: str = "data/raw",
    output_dir: str = "data/processed",
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
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    segmenter = config['preprocessing']['segmenter']
    lowercase = config['preprocessing'].get('lowercase', False)

    # Initialize preprocessor
    preprocessor = VietnamesePreprocessor(segmenter=segmenter)
    logger.info(f"Using segmenter: {segmenter}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split_name in ['train', 'validation', 'test']:
        input_file = Path(input_dir) / f"{split_name}.csv"

        if not input_file.exists():
            logger.warning(f"{input_file} not found. Skipping...")
            continue

        logger.info(f"Processing {split_name} split...")

        # Load data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} samples")

        # Detect text column
        text_col = None
        for col in ['text', 'sentence', 'content']:
            if col in df.columns:
                text_col = col
                break

        if text_col is None:
            # Use first non-label column
            label_cols = ['label', 'labels', 'emotion']
            text_col = [col for col in df.columns if col not in label_cols][0]

        logger.info(f"Using column '{text_col}' for text")

        # Detect label column (case-insensitive)
        label_col = None
        for col in df.columns:
            if col.lower() in ['label', 'labels', 'emotion']:
                label_col = col
                break

        if label_col is None:
            label_col = df.columns[-1]

        logger.info(f"Using column '{label_col}' for labels")

        # Build label mapping from config (invert: "Enjoyment" -> 0)
        emotion_labels = config.get('emotion_labels', {})
        label_to_int = {v: int(k) for k, v in emotion_labels.items()}

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
        default="data/merged",
        help="Directory containing raw CSV files (default: data/merged)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to save preprocessed files (default: data/processed)"
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
