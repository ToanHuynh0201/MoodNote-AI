"""
PyTorch Dataset class for UIT-VSMEC
"""
from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ...utils.logger import get_logger

logger = get_logger("dataset")


class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion classification"""

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "vinai/phobert-base-v2",
        max_length: int = 128,
        tokenizer=None,
    ) -> None:
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
        logger.info(f"Loaded {len(self.df)} samples from {data_path}")

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = tokenizer

        # Extract texts and labels
        texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()

        # Pre-tokenize entire dataset once to avoid repeated tokenization per epoch
        logger.info(f"Tokenizing {len(texts)} samples...")
        encodings = self.tokenizer(
            [str(t) for t in texts],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer_name: str = "vinai/phobert-base-v2",
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0,
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
    _pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin
    )

    logger.info(
        f"DataLoaders created: "
        f"train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader, tokenizer
