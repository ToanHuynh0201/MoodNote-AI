"""
Vietnamese text preprocessing with word segmentation for PhoBERT
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml

try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False
    print("Warning: pyvi not installed. Install with: pip install pyvi")


class VietnamesePreprocessor:
    """Vietnamese text preprocessor with word segmentation"""

    def __init__(self, segmenter="pyvi"):
        """
        Initialize preprocessor

        Args:
            segmenter: Type of segmenter to use ("pyvi" or "vncorenlp")
        """
        self.segmenter = segmenter

        if segmenter == "pyvi" and not PYVI_AVAILABLE:
            raise ImportError("pyvi is not installed. Install with: pip install pyvi")

    def segment_text(self, text):
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
            segmented = ViTokenizer.tokenize(text)
            return segmented
        else:
            raise ValueError(f"Unsupported segmenter: {self.segmenter}")

    def preprocess_text(self, text, lowercase=False):
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
    input_dir="data/raw",
    output_dir="data/processed",
    config_path="configs/model_config.yaml"
):
    """
    Preprocess UIT-VSMEC dataset with Vietnamese word segmentation

    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory to save preprocessed files
        config_path: Path to model configuration file
    """
    print("Starting Vietnamese text preprocessing...")

    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    segmenter = config['preprocessing']['segmenter']
    lowercase = config['preprocessing'].get('lowercase', False)

    # Initialize preprocessor
    preprocessor = VietnamesePreprocessor(segmenter=segmenter)
    print(f"Using segmenter: {segmenter}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split_name in ['train', 'validation', 'test']:
        input_file = Path(input_dir) / f"{split_name}.csv"

        if not input_file.exists():
            print(f"Warning: {input_file} not found. Skipping...")
            continue

        print(f"\nProcessing {split_name} split...")

        # Load data
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} samples")

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

        print(f"Using column '{text_col}' for text")

        # Detect label column (case-insensitive)
        label_col = None
        for col in df.columns:
            if col.lower() in ['label', 'labels', 'emotion']:
                label_col = col
                break

        if label_col is None:
            label_col = df.columns[-1]

        print(f"Using column '{label_col}' for labels")

        # Build label mapping from config (invert: "Enjoyment" -> 0)
        emotion_labels = config.get('emotion_labels', {})
        label_to_int = {v: int(k) for k, v in emotion_labels.items()}

        # Preprocess texts
        print("Applying word segmentation...")
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
        print(f"Saved preprocessed data to {output_file}")

        # Show examples
        print("\nExamples:")
        for i in range(min(3, len(df))):
            print(f"\nOriginal: {df[text_col].iloc[i]}")
            print(f"Segmented: {processed_df['text'].iloc[i]}")
            print(f"Label: {processed_df['label'].iloc[i]}")

    print("\n✓ Preprocessing complete!")


def main():
    """Main function"""
    preprocess_dataset()


if __name__ == "__main__":
    main()
