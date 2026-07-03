from __future__ import annotations

from torch.utils.data import DataLoader

from src.client.base_client import FlowerClient
from src.client.dp_client import DPClient
from src.config.loader import ExperimentConfig
from src.models.base import BaseModel


def create_client(
    cid: int,
    model: BaseModel,
    trainloader: DataLoader,
    valloader: DataLoader,
    config: ExperimentConfig,
) -> FlowerClient:
    if config.privacy.enabled:
        return DPClient(model, trainloader, valloader, config)
    return FlowerClient(model, trainloader, valloader, config)
