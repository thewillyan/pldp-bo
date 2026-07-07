from __future__ import annotations

from src.config.loader import ModelConfig
from src.models.base import BaseModel
from src.models.cnn import CNNModel
from src.models.mlp import MLPModel

_MODEL_DATA_COMPAT: dict[str, list[str]] = {
    "cnn": ["cifar10", "cifar100"],
    "mlp": ["mnist", "cifar10", "cifar100"],
}


def create_model(config: ModelConfig, dataset_name: str | None = None) -> BaseModel:
    if dataset_name is not None:
        allowed = _MODEL_DATA_COMPAT.get(config.name)
        if allowed and dataset_name not in allowed:
            raise ValueError(
                f"Model '{config.name}' is not compatible with dataset '{dataset_name}'. "
                f"Compatible datasets: {allowed}"
            )
    if config.name == "cnn":
        return CNNModel(config.num_classes)
    elif config.name == "mlp":
        return MLPModel(config.num_classes)
    else:
        raise ValueError(f"Unknown model: {config.name}")
