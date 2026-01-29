"""
Download UIT-VSMEC dataset from Hugging Face
"""
import os
from datasets import load_dataset
import pandas as pd
from pathlib import Path


def download_uit_vsmec(output_dir="data/raw"):
    """
    Download UIT-VSMEC dataset from Hugging Face and save to CSV files.

    Args:
        output_dir: Directory to save the downloaded dataset

    Returns:
        dict: Dictionary containing train, validation, and test DataFrames
    """
    print("Downloading UIT-VSMEC dataset from Hugging Face...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load dataset from Hugging Face
        dataset = load_dataset("tridm/UIT-VSMEC")

        print(f"Dataset loaded successfully!")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['validation'])}")
        print(f"Test samples: {len(dataset['test'])}")

        # Convert to pandas DataFrames and save
        splits = {}
        for split_name in ['train', 'validation', 'test']:
            # Convert to DataFrame
            df = dataset[split_name].to_pandas()
            splits[split_name] = df

            # Save to CSV
            output_file = output_path / f"{split_name}.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"Saved {split_name} split to {output_file}")

            # Print sample
            print(f"\nSample from {split_name}:")
            print(df.head(2))
            print()

        # Print emotion distribution
        print("\nEmotion distribution in training set:")
        emotion_labels = {
            0: "Enjoyment",
            1: "Sadness",
            2: "Anger",
            3: "Fear",
            4: "Disgust",
            5: "Surprise",
            6: "Other"
        }

        label_col = 'label' if 'label' in splits['train'].columns else 'labels'
        emotion_counts = splits['train'][label_col].value_counts().sort_index()

        for label, count in emotion_counts.items():
            emotion_name = emotion_labels.get(label, f"Unknown_{label}")
            percentage = (count / len(splits['train'])) * 100
            print(f"{emotion_name}: {count} ({percentage:.2f}%)")

        print("\nDataset download complete!")
        return splits

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please make sure you have internet connection and the dataset is accessible.")
        raise


def main():
    """Main function to download dataset"""
    download_uit_vsmec()


if __name__ == "__main__":
    main()
