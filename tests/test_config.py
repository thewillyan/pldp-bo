from __future__ import annotations

from pathlib import Path

import yaml

from src.config.loader import ExperimentConfig, load_config


def test_load_default_config() -> None:
    config = load_config("config/default.yaml")
    assert config.data.name == "cifar10"
    assert config.federated.num_rounds == 50
    assert config.privacy.enabled is False


def test_load_experiment_config() -> None:
    config = load_config("config/dp_example.yaml")
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


def test_load_personalization_config() -> None:
    config = load_config("config/personalized_custom.yaml")
    assert config.personalization.enabled is True
    assert config.personalization.strategy == "custom"
    assert config.personalization.client_epsilon_map[0] == 1.0
    assert config.personalization.track_cumulative is True


def test_personalization_defaults_when_absent() -> None:
    config = load_config("config/default.yaml")
    assert config.personalization.enabled is False
    assert config.personalization.strategy == "uniform"
    assert config.personalization.epsilon_min == 0.1
    assert config.personalization.epsilon_max == 10.0
    assert config.personalization.client_epsilon_map == {}


def test_personalization_config_override() -> None:
    config = load_config(
        "config/default.yaml",
        overrides={"personalization.enabled": True, "personalization.strategy": "heterogeneity"},
    )
    assert config.personalization.enabled is True
    assert config.personalization.strategy == "heterogeneity"


def test_bo_config_defaults() -> None:
    config = load_config("config/default.yaml")
    assert config.bo.enabled is False
    assert config.bo.warmup_rounds == 20
    assert config.bo.epsilon_budget == 10.0
    assert config.bo.optimization_metric == "nun"


def test_bo_config_override() -> None:
    config = load_config(
        "config/default.yaml",
        overrides={
            "bo.enabled": True,
            "bo.warmup_rounds": 10,
            "bo.epsilon_budget": 5.0,
            "bo.optimization_metric": "utility",
        },
    )
    assert config.bo.enabled is True
    assert config.bo.warmup_rounds == 10
    assert config.bo.epsilon_budget == 5.0
    assert config.bo.optimization_metric == "utility"
