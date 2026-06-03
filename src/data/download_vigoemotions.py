"""
Download ViGoEmotions dataset from Hugging Face.

ViGoEmotions is a Vietnamese multi-label emotion dataset with 27 fine-grained
emotion categories (~20,664 samples). This script downloads and normalizes it
to a consistent format for merging with UIT-VSMEC.

Output format: CSV with columns:
  - text: Vietnamese sentence
  - labels: JSON string of fine-grained emotion label(s), e.g. '["joy", "admiration"]'
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from ..utils.logger import get_logger

logger = get_logger("download_vigoemotions")

# All 27 fine-grained emotion labels in ViGoEmotions
VIGOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride",
    "realization", "relief", "remorse", "sadness", "surprise",
]


def detect_label_format(df: pd.DataFrame) -> str:
    """
    Detect how labels are stored in the DataFrame.

    Returns:
        "list_column"    — has a 'labels' column containing lists of strings
        "binary_columns" — has individual boolean/int columns per emotion
    """
    if "labels" in df.columns:
        sample = df["labels"].iloc[0]
        if isinstance(sample, (list, str)):
            return "list_column"

    # Check if individual emotion columns exist
    emotion_cols = [col for col in df.columns if col in VIGOEMOTIONS_LABELS]
    if len(emotion_cols) >= 5:
        return "binary_columns"

    # Fallback: assume list_column and let downstream handle errors
    return "list_column"


def normalize_to_list_format(df: pd.DataFrame, format_type: str) -> pd.DataFrame:
    """
    Normalize any ViGoEmotions format to a uniform DataFrame with:
      - 'text': the Vietnamese sentence
      - 'labels': Python list of fine-grained emotion strings

    Args:
        df: Raw DataFrame from HuggingFace
        format_type: "list_column" or "binary_columns"

    Returns:
        Normalized DataFrame with columns ['text', 'labels'] (labels as Python list)
    """
    # Detect text column
    text_col = None
    for candidate in ["text", "sentence", "content", "comment_text"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        # Use first non-label column
        label_cols = set(VIGOEMOTIONS_LABELS) | {"labels", "id"}
        text_col = next(c for c in df.columns if c not in label_cols)

    if format_type == "list_column":
        labels_raw = df["labels"].tolist()
        labels_parsed = []
        for item in labels_raw:
            if isinstance(item, list):
                # Convert int indices → string names if needed
                resolved = []
                for v in item:
                    if isinstance(v, int):
                        if 0 <= v < len(VIGOEMOTIONS_LABELS):
                            resolved.append(VIGOEMOTIONS_LABELS[v])
                        # else: out-of-range index, skip
                    else:
                        resolved.append(str(v))
                labels_parsed.append(resolved)
            elif isinstance(item, str):
                # Try JSON parse, then comma-split
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, list):
                        resolved = []
                        for v in parsed:
                            if isinstance(v, int):
                                if 0 <= v < len(VIGOEMOTIONS_LABELS):
                                    resolved.append(VIGOEMOTIONS_LABELS[v])
                            else:
                                resolved.append(str(v))
                        labels_parsed.append(resolved)
                    else:
                        labels_parsed.append([str(parsed)])
                except json.JSONDecodeError:
                    labels_parsed.append([s.strip() for s in item.split(",") if s.strip()])
            else:
                labels_parsed.append([])

    elif format_type == "binary_columns":
        emotion_cols = [col for col in df.columns if col in VIGOEMOTIONS_LABELS]
        labels_parsed = []
        for _, row in df.iterrows():
            active = [col for col in emotion_cols if row[col]]
            labels_parsed.append(active)

    else:
        raise ValueError(f"Unknown format_type: {format_type}")

    return pd.DataFrame({"text": df[text_col].tolist(), "labels": labels_parsed})


def download_vigoemotions(output_dir: str = "data/raw", token: str | None = None) -> dict:
    """
    Download uitnlp/vigoemotions from HuggingFace and save to CSV files.

    NOTE: This is a gated dataset. You must be authenticated before calling this.
    Options:
      1. Run `huggingface-cli login` once in your terminal, OR
      2. Pass token= parameter directly, OR
      3. Set HF_TOKEN environment variable.

    Saves to: {output_dir}/vigoemotions/{train,validation,test}.csv
    Each row: text (str), labels (JSON string, e.g. '["joy"]')

    Args:
        output_dir: Base raw data directory (default: data/raw)
        token: HuggingFace access token (optional, reads from env/cache if None)

    Returns:
        dict mapping split name → normalized DataFrame (labels as Python list)
    """
    import os
    hf_token = token or os.environ.get("HF_TOKEN")

    logger.info("Downloading ViGoEmotions dataset from Hugging Face...")
    if hf_token:
        logger.info("  Using provided HuggingFace token.")
    else:
        logger.info("  No token provided — using cached login (run `huggingface-cli login` if needed).")

    output_path = Path(output_dir) / "vigoemotions"
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset("uitnlp/vigoemotions", token=hf_token)
    except Exception as e:
        if "gated" in str(e).lower() or "authenticated" in str(e).lower():
            logger.error("This is a gated dataset. To access it:")
            logger.error("  Option 1: huggingface-cli login")
            logger.error("  Option 2: python -m src.data.download_vigoemotions --token YOUR_TOKEN")
            logger.error("  Option 3: set HF_TOKEN=YOUR_TOKEN in environment")
            logger.error("Get your token at: https://huggingface.co/settings/tokens")
        else:
            logger.error(f"Error downloading dataset: {e}")
        raise

    logger.info("Dataset loaded successfully!")

    splits = {}
    all_label_counts: Counter = Counter()
    total_labels = 0
    total_samples = 0

    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            logger.warning(f"Split '{split_name}' not found, skipping.")
            continue

        raw_df: pd.DataFrame = dataset[split_name].to_pandas()  # type: ignore[assignment]
        logger.info(f"{split_name}: {len(raw_df)} samples | columns: {list(raw_df.columns)}")

        fmt = detect_label_format(raw_df)
        logger.info(f"  Detected label format: {fmt}")

        norm_df = normalize_to_list_format(raw_df, fmt)
        splits[split_name] = norm_df

        # Serialize labels list → JSON string for CSV storage
        csv_df = norm_df.copy()
        csv_df["labels"] = csv_df["labels"].apply(json.dumps, ensure_ascii=False)

        output_file = output_path / f"{split_name}.csv"
        csv_df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"  Saved to {output_file}")

        # Accumulate stats
        for label_list in norm_df["labels"]:
            all_label_counts.update(label_list)
            total_labels += len(label_list)
        total_samples += len(norm_df)

        # Show sample
        sample = norm_df.iloc[0]
        logger.debug(f"  Sample row: text={sample['text'][:80]} | labels={sample['labels']}")

    # Summary stats
    logger.info(f"Total samples: {total_samples}")
    if total_samples:
        logger.info(f"Avg labels per sample: {total_labels / total_samples:.2f}")
    logger.info("Top 15 most common fine-grained labels (across all splits):")
    for label, count in all_label_counts.most_common(15):
        logger.info(f"  {str(label):20s}: {count:5d}")

    logger.info("ViGoEmotions download complete!")
    return splits


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Download ViGoEmotions dataset from HuggingFace")
    parser.add_argument(
        "--output-dir", default="data/raw",
        help="Base output directory (default: data/raw)"
    )
    parser.add_argument(
        "--token", default=None,
        help="HuggingFace access token (or set HF_TOKEN env var, or run huggingface-cli login)"
    )
    args = parser.parse_args()
    download_vigoemotions(output_dir=args.output_dir, token=args.token)


if __name__ == "__main__":
    main()
