"""
Model utilities for loading, saving, and managing models
"""
import torch
from pathlib import Path
from transformers import AutoTokenizer
from .phobert_classifier import PhoBERTEmotionClassifier


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
        'hidden_size': model.hidden_size
    }, model_path)

    # Save tokenizer
    tokenizer.save_pretrained(save_path)

    # Save config if provided
    if config is not None:
        import yaml
        config_path = save_path / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

    print(f"Model saved to {save_dir}")


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

    # Create model
    model = PhoBERTEmotionClassifier(
        model_name=checkpoint['model_name'],
        num_labels=checkpoint['num_labels']
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(load_path)

    print(f"Model loaded from {load_dir}")
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
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

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


if __name__ == "__main__":
    # Test model utils
    from .phobert_classifier import PhoBERTEmotionClassifier

    print("Testing model utilities...")

    # Create model
    model = PhoBERTEmotionClassifier(
        model_name="vinai/phobert-base",
        num_labels=7
    )

    # Print summary
    print_model_summary(model)

    # Get device
    device = get_device()

    print("\nParameter counts:")
    params = count_parameters(model)
    for key, value in params.items():
        print(f"{key}: {value:,}")
