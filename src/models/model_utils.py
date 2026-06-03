"""
Model utilities for loading, saving, and managing models
"""
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer
from .phobert_classifier import PhoBERTEmotionClassifier
from ..utils.logger import get_logger


logger = get_logger("model_utils")


def save_model(model, tokenizer, save_dir, config=None):
    """
    Save model, tokenizer, and configuration

    Args:
        model: PhoBERT model
        tokenizer: Tokenizer
        save_dir: Directory to save model
        config: Optional configuration dictionary
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    model_path = save_path / "model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model.model_name,
        'num_labels': model.num_labels,
        'hidden_size': model.hidden_size,
        'dropout': model.dropout.p,
        'label_smoothing': model.label_smoothing,
        'focal_gamma': model.focal_gamma
    }, model_path)

    # Save tokenizer
    tokenizer.save_pretrained(save_path)

    # Save config if provided
    if config is not None:
        import yaml
        config_path = save_path / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

    logger.info(f"Model saved to {save_dir}")


def load_model(load_dir, device='cpu'):
    """
    Load model and tokenizer

    Args:
        load_dir: Directory containing saved model
        device: Device to load model on

    Returns:
        tuple: (model, tokenizer)
    """
    load_path = Path(load_dir)

    # Load model checkpoint
    model_path = load_path / "model.pt"
    checkpoint = torch.load(model_path, map_location=device)

    # Create model. Prefer offline/local cache to avoid unnecessary HF Hub calls at API startup.
    model_name = checkpoint['model_name']
    model_kwargs = {
        'model_name': model_name,
        'num_labels': checkpoint['num_labels'],
        'dropout': checkpoint.get('dropout', 0.1),
        'label_smoothing': checkpoint.get('label_smoothing', 0.0),
        'focal_gamma': checkpoint.get('focal_gamma', 0.0)
    }

    allow_hf_fallback = os.getenv("MOODNOTE_ALLOW_HF_FALLBACK", "0") == "1"
    try:
        model = PhoBERTEmotionClassifier(
            **model_kwargs,
            local_files_only=True
        )
    except Exception as exc:
        if not allow_hf_fallback:
            raise RuntimeError(
                f"Local/cached files for {model_name} are unavailable ({exc}). "
                "Set MOODNOTE_ALLOW_HF_FALLBACK=1 to allow downloading from Hugging Face Hub."
            ) from exc

        logger.warning(
            f"Local/cached files for {model_name} are unavailable ({exc}). "
            "Falling back to Hugging Face Hub because MOODNOTE_ALLOW_HF_FALLBACK=1."
        )
        model = PhoBERTEmotionClassifier(
            **model_kwargs,
            local_files_only=False
        )

    # Load state dict — strict=False để tương thích checkpoint cũ có bert.pooler.*
    # (pooler bị loại khỏi model vì dùng mean pooling, không phải CLS pooling)
    result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    unexpected = [k for k in result.unexpected_keys if not k.startswith('bert.pooler')]
    missing = [k for k in result.missing_keys if not k.startswith('bert.pooler')]
    if unexpected:
        logger.warning(f"Unexpected keys khi load model: {unexpected}")
    if missing:
        logger.warning(f"Missing keys khi load model: {missing}")
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=True)

    logger.info(f"Model loaded from {load_dir}")
    return model, tokenizer


def count_parameters(model):
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        dict: Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def get_device():
    """
    Get available device (CUDA if available, else CPU)

    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    return device


def print_model_summary(model):
    """
    Print model summary

    Args:
        model: PyTorch model
    """
    print("\n" + "=" * 50)
    print("Model Summary")
    print("=" * 50)

    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters: {params['frozen']:,}")

    print("\nModel architecture:")
    print(model)
    print("=" * 50 + "\n")
