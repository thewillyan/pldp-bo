from __future__ import annotations

import contextlib
import logging
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    name: str = "cifar10"
    data_dir: str = "./data"
    num_clients: int = 10
    partition_type: str = "iid"
    partition_alpha: float = 1.0
    batch_size: int = 64
    val_split: float = 0.1


@dataclass
class ModelConfig:
    name: str = "cnn"
    num_classes: int = 10


@dataclass
class FederatedConfig:
    num_rounds: int = 50
    fraction_fit: float = 0.5
    fraction_evaluate: float = 0.2
    local_epochs: int = 5
    strategy: str = "fedavg"
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_nodes: int = 2
    proximal_mu: float = 0.0
    server_learning_rate: float = 1.0


@dataclass
class OptimizerConfig:
    name: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0


@dataclass
class PrivacyConfig:
    enabled: bool = False
    mechanism: str = "gaussian"
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    target_epsilon: float | None = None
    accountant: str = "rdp"
    total_budget: float | None = None


@dataclass
class PersonalizationConfig:
    enabled: bool = False
    strategy: str = "uniform"
    epsilon_min: float = 0.1
    epsilon_max: float = 10.0
    epsilon_base: float = 1.0
    client_epsilon_map: dict = field(default_factory=dict)
    track_cumulative: bool = True


@dataclass
class BOConfig:
    enabled: bool = False
    warmup_rounds: int = 20
    epsilon_min: float = 0.1
    epsilon_max: float = 10.0
    epsilon_budget: float = 10.0
    optimization_metric: str = "nun"
    grid_points: int = 100
    acquisition_penalty: float = 0.1
    gp_kernel: str = "matern52"
    observation_noise: float = 0.01

    # Per-client bounds personalization
    bounds_strategy: str = "global"
    bounds_ratio_min: float = 0.1
    bounds_ratio_max: float = 1.0
    client_eps_min_map: dict = field(default_factory=dict)
    client_eps_max_map: dict = field(default_factory=dict)
    client_warmup_rounds_map: dict = field(default_factory=dict)


@dataclass
class LoggingConfig:
    tracker: str = "mlflow"
    experiment_name: str = "pldp-bo"
    run_name: str | None = None
    tracking_uri: str = "./mlruns"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    personalization: PersonalizationConfig = field(default_factory=PersonalizationConfig)
    bo: BOConfig = field(default_factory=BOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    deterministic: bool = False


_CONFIG_KEY_MAP = {
    "data": DataConfig,
    "model": ModelConfig,
    "federated": FederatedConfig,
    "optimizer": OptimizerConfig,
    "privacy": PrivacyConfig,
    "personalization": PersonalizationConfig,
    "bo": BOConfig,
    "logging": LoggingConfig,
}


def _expand_dot_keys(overrides: dict) -> dict:
    result: dict = {}
    for key, value in overrides.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result


def _merge_dict_into_dataclass(dc_instance: object, override: dict) -> None:
    for key, value in override.items():
        if hasattr(dc_instance, key):
            if isinstance(value, dict) and hasattr(getattr(dc_instance, key), "__dataclass_fields__"):
                _merge_dict_into_dataclass(getattr(dc_instance, key), value)
            else:
                setattr(dc_instance, key, value)
        else:
            logger.warning("Unknown config override key: '%s'", key)


def load_config(config_path: str, overrides: dict | None = None) -> ExperimentConfig:
    config = ExperimentConfig()

    path = Path(config_path)
    if path.exists():
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
        for key, sub_config in raw.items():
            if key in _CONFIG_KEY_MAP and isinstance(sub_config, dict):
                dc_type = _CONFIG_KEY_MAP[key]
                current = getattr(config, key)
                type_hints = typing.get_type_hints(dc_type)
                for fld in fields(dc_type):
                    if fld.name in sub_config:
                        value = sub_config[fld.name]
                        expected = type_hints.get(fld.name)
                        if isinstance(value, str) and expected in (float, int):
                            with contextlib.suppress(ValueError, TypeError):
                                value = expected(value)
                        setattr(current, fld.name, value)

    if overrides:
        expanded = _expand_dot_keys(overrides)
        _merge_dict_into_dataclass(config, expanded)

    return config
