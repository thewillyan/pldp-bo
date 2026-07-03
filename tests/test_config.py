from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from src.config.loader import ExperimentConfig, load_config


def test_load_default_config() -> None:
    config = load_config("config/default.yaml")
    assert config.data.name == "cifar10"
    assert config.federated.num_rounds == 50
    assert config.privacy.enabled is False


def test_load_experiment_config() -> None:
    config = load_config("config/experiments/dp_example.yaml")
    assert config.privacy.enabled is True
    assert config.privacy.noise_multiplier == 1.0


def test_config_override() -> None:
    config = load_config("config/default.yaml", overrides={"data.num_clients": 5})
    assert config.data.num_clients == 5


def test_config_from_empty_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")
    config = load_config(str(config_path))
    assert isinstance(config, ExperimentConfig)


def test_config_from_custom_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(yaml.dump({
        "data": {"name": "mnist", "num_clients": 3},
        "federated": {"num_rounds": 5},
    }))
    config = load_config(str(config_path))
    assert config.data.name == "mnist"
    assert config.federated.num_rounds == 5
    assert config.model.name == "cnn"  # default preserved
