from __future__ import annotations

import logging
import types
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
    update_clip_norm: float = 1.0
    delta: float = 1e-5
    target_epsilon: float | None = None
    accountant: str = "rdp"
    total_budget: float | None = None  # total epsilon budget across ALL clients (divided per-client)


@dataclass
class PersonalizationConfig:
    enabled: bool = False
    strategy: str = "uniform"
    client_epsilon_map: dict = field(default_factory=dict)
    track_cumulative: bool = True


@dataclass
class BOConfig:
    enabled: bool = False
    max_warmup_ratio: float = 0.0
    min_warmup: int = 3
    epsilon_min: float = 0.2
    epsilon_max: float = 10.0
    epsilon_budget: float = 10.0
    optimization_metric: str = "nun"
    grid_points: int = 100
    acquisition_penalty: float = 0.1
    gp_kernel: str = "matern52"
    observation_noise: float = 0.01
    budget_margin: float = 0.1  # fraction of remaining budget reserved when masking grid points

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
    group: str | None = None


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
        for depth, part in enumerate(parts[:-1]):
            if part in current and not isinstance(current[part], dict):
                logger.warning(
                    "Config override key conflict: '%s' overlaps with existing key at '%s'",
                    key, ".".join(parts[:depth + 1]),
                )
                break
            current = current.setdefault(part, {})
        else:
            existing = current.get(parts[-1])
            if isinstance(existing, dict):
                logger.warning(
                    "Config override key conflict: '%s' would overwrite nested config",
                    key,
                )
            current[parts[-1]] = value
    return result


def _unwrap_optional(tp: type) -> type:
    """If tp is Optional[X] (i.e. X | None), return X. Otherwise return tp unchanged."""
    origin = typing.get_origin(tp)
    if origin is types.UnionType:
        args = typing.get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return tp


def _coerce_value(value: object, expected: type) -> object:
    if not isinstance(value, str):
        return value
    expected = _unwrap_optional(expected)
    if expected is bool:
        return value.lower() in ("true", "1", "yes")
    if expected is int:
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning("Cannot convert override value '%s' to int", value)
            return value
    if expected is float:
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning("Cannot convert override value '%s' to float", value)
            return value
    return value


def _merge_dict_into_dataclass(
    dc_instance: object, override: dict, dc_type: type | None = None,
) -> None:
    if dc_type is None:
        dc_type = type(dc_instance)
    type_hints = typing.get_type_hints(dc_type)
    for key, value in override.items():
        if not hasattr(dc_instance, key):
            logger.warning("Unknown config override key: '%s'", key)
            continue
        if isinstance(value, dict) and hasattr(getattr(dc_instance, key), "__dataclass_fields__"):
            child_type = type(getattr(dc_instance, key))
            _merge_dict_into_dataclass(getattr(dc_instance, key), value, dc_type=child_type)
        else:
            expected = type_hints.get(key)
            if expected is not None:
                value = _coerce_value(value, expected)
            setattr(dc_instance, key, value)


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
                            try:
                                value = expected(value)
                            except (ValueError, TypeError):
                                logger.warning(
                                    "Config key '%s.%s': cannot convert '%s' to %s",
                                    key, fld.name, value, expected.__name__,
                                )
                        setattr(current, fld.name, value)

    if overrides:
        expanded = _expand_dot_keys(overrides)
        _merge_dict_into_dataclass(config, expanded)

    return config
