"""Tests for pydantic config-schema validation."""

from src.utils.config_schema import validate_configs


def test_validate_configs_passes_for_real_config_files():
    result = validate_configs()

    assert set(result) == {"model", "training", "api"}
    assert result["model"].model.num_labels == 7
    assert result["training"].training.batch_size == 16
    assert result["api"].api.port == 8000
