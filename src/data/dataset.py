"""
PyTorch Dataset class for UIT-VSMEC
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path


class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion classification"""

    def __init__(
        self,
        data_path,
        tokenizer_name="vinai/phobert-base",
        max_length=128,
        tokenizer=None
    ):
        """
        Initialize dataset

        Args:
            data_path: Path to preprocessed CSV file
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
            tokenizer: Pre-initialized tokenizer (optional)
        """
        self.data_path = data_path
        self.max_length = max_length

        # Load data
        self.df = pd.read_csv(data_path)
        print(f"Loaded {len(self.df)} samples from {data_path}")

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = tokenizer

        # Extract texts and labels
        self.texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()

    def __len__(self):
        """Return dataset size"""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get item by index

        Args:
            idx: Index

        Returns:
            dict: Dictionary containing input_ids, attention_mask, and label
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(
    train_path,
    val_path,
    test_path,
    tokenizer_name="vinai/phobert-base",
    batch_size=16,
    max_length=128,
    num_workers=0
):
    """
    Create DataLoaders for train, validation, and test sets

    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        tokenizer_name: Name of tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for DataLoader

    Returns:
        tuple: (train_loader, val_loader, test_loader, tokenizer)
    """
    from torch.utils.data import DataLoader

    # Initialize tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create datasets
    train_dataset = EmotionDataset(
        train_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        tokenizer=tokenizer
    )

    val_dataset = EmotionDataset(
        val_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        tokenizer=tokenizer
    )

    test_dataset = EmotionDataset(
        test_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        tokenizer=tokenizer
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"\nDataLoaders created:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, tokenizer


if __name__ == "__main__":
    # Test dataset
    dataset = EmotionDataset(
        data_path="data/processed/train.csv",
        max_length=128
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"\nSample item:")
    sample = dataset[0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Label: {sample['labels']}")
