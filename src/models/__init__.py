from __future__ import annotations

from src.config.loader import ModelConfig
from src.models.base import BaseModel
from src.models.cnn import CNNModel
from src.models.mlp import MLPModel


def create_model(config: ModelConfig) -> BaseModel:
    if config.name == "cnn":
        return CNNModel(config.num_classes)
    elif config.name == "mlp":
        return MLPModel(config.num_classes)
    else:
        raise ValueError(f"Unknown model: {config.name}")
