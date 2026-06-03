"""
Download UIT-VSMEC dataset from Hugging Face
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from ..utils.emotion_constants import DEFAULT_EMOTION_LABELS
from ..utils.logger import get_logger

logger = get_logger("download_dataset")


def download_uit_vsmec(output_dir: str = "data/raw") -> dict:
    """
    Download UIT-VSMEC dataset from Hugging Face and save to CSV files.

    Args:
        output_dir: Directory to save the downloaded dataset

    Returns:
        dict: Dictionary containing train, validation, and test DataFrames
    """
    logger.info("Downloading UIT-VSMEC dataset from Hugging Face...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load dataset from Hugging Face
        dataset = load_dataset("tridm/UIT-VSMEC")

        logger.info("Dataset loaded successfully!")
        logger.info(f"Train samples: {len(dataset['train'])}")
        logger.info(f"Validation samples: {len(dataset['validation'])}")
        logger.info(f"Test samples: {len(dataset['test'])}")

        # Convert to pandas DataFrames and save
        splits = {}
        for split_name in ['train', 'validation', 'test']:
            # Convert to DataFrame
            df: pd.DataFrame = dataset[split_name].to_pandas()  # type: ignore[assignment]
            splits[split_name] = df

            # Save to CSV
            output_file = output_path / f"{split_name}.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Saved {split_name} split to {output_file}")

        # Log emotion distribution
        logger.info("Emotion distribution in training set:")
        train_df = splits['train']
        if 'Emotion' in train_df.columns:
            emotion_counts = train_df['Emotion'].value_counts().sort_index()
            for emotion_name, count in emotion_counts.items():
                percentage = (count / len(train_df)) * 100
                logger.info(f"  {emotion_name}: {count} ({percentage:.2f}%)")
        else:
            label_col = next((c for c in ('label', 'labels') if c in train_df.columns), None)
            if label_col is None:
                logger.warning(f"Unknown column layout: {list(train_df.columns)}")
            else:
                emotion_counts = train_df[label_col].value_counts().sort_index()
                for label, count in emotion_counts.items():
                    emotion_name = DEFAULT_EMOTION_LABELS.get(label, f"Unknown_{label}")
                    percentage = (count / len(train_df)) * 100
                    logger.info(f"  {emotion_name}: {count} ({percentage:.2f}%)")

        logger.info("Dataset download complete!")
        return splits

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.error("Ensure you have internet connection and the dataset is accessible.")
        raise


def main() -> None:
    """Main function to download dataset"""
    download_uit_vsmec()


if __name__ == "__main__":
    main()
