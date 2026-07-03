from __future__ import annotations

from torch.utils.data import DataLoader

from src.client.base_client import FlowerClient
from src.client.dp_client import DPClient
from src.config.loader import ExperimentConfig
from src.models.base import BaseModel
from src.privacy.accountant import RDPAccountant


def create_client(
    cid: int,
    model: BaseModel,
    trainloader: DataLoader,
    valloader: DataLoader,
    config: ExperimentConfig,
    client_epsilon: float | None = None,
    accountant: RDPAccountant | None = None,
    num_rounds: int | None = None,
) -> FlowerClient:
    if config.privacy.enabled:
        return DPClient(
            model,
            trainloader,
            valloader,
            config,
            client_epsilon=client_epsilon,
            accountant=accountant,
            num_rounds=num_rounds,
        )
    return FlowerClient(model, trainloader, valloader, config)
