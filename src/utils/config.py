"""
Configuration management utilities
"""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file

    Args:
        config_path: Path to YAML config file

    Returns:
        dict: Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def load_all_configs(config_dir: str = "configs") -> Dict[str, Dict[str, Any]]:
    """
    Load all configuration files

    Args:
        config_dir: Directory containing config files

    Returns:
        dict: Dictionary containing all configs
    """
    config_path = Path(config_dir)

    configs = {
        'model': load_config(config_path / "model_config.yaml"),
        'training': load_config(config_path / "training_config.yaml"),
        'api': load_config(config_path / "api_config.yaml")
    }

    return configs


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Config saved to {config_path}")


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries

    Args:
        *configs: Variable number of config dictionaries

    Returns:
        dict: Merged configuration
    """
    merged = {}

    for config in configs:
        merged.update(config)

    return merged


def get_config_value(config: Dict[str, Any], key_path: str, default=None):
    """
    Get nested config value using dot notation

    Args:
        config: Configuration dictionary
        key_path: Path to value (e.g., "model.name")
        default: Default value if key not found

    Returns:
        Value at key_path or default
    """
    keys = key_path.split('.')
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


if __name__ == "__main__":
    # Test config loading
    print("Testing configuration utilities...")

    # Load all configs
    configs = load_all_configs()

    print("\nModel config:")
    print(yaml.dump(configs['model'], default_flow_style=False))

    print("\nTraining config:")
    print(yaml.dump(configs['training'], default_flow_style=False))

    # Test get_config_value
    model_name = get_config_value(configs, 'model.model.name')
    print(f"\nModel name: {model_name}")

    learning_rate = get_config_value(configs, 'training.training.learning_rate')
    print(f"Learning rate: {learning_rate}")
