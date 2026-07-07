"""Tests for the YAML config loading utilities."""

import pytest

from src.utils.config import get_config_value, load_config


def test_load_config_reads_yaml(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("model:\n  name: phobert\n  num_labels: 7\n", encoding="utf-8")

    config = load_config(cfg_file)

    assert config["model"]["name"] == "phobert"
    assert config["model"]["num_labels"] == 7


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_get_config_value_dot_notation():
    config = {"model": {"name": "phobert", "params": {"dropout": 0.4}}}
    assert get_config_value(config, "model.name") == "phobert"
    assert get_config_value(config, "model.params.dropout") == 0.4


def test_get_config_value_returns_default_when_missing():
    config = {"model": {"name": "phobert"}}
    assert get_config_value(config, "model.unknown", default="fallback") == "fallback"
    assert get_config_value(config, "a.b.c") is None
