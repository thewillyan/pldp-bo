from __future__ import annotations

from torch.utils.data import DataLoader

from src.client.base_client import FlowerClient
from src.client.per_update_dp_client import PerUpdateDPClient
from src.config.loader import ExperimentConfig
from src.models.base import BaseModel
from src.privacy.accountant import RDPAccountant


def create_client(
    cid: int,  # noqa: ARG001
    model: BaseModel,
    trainloader: DataLoader,
    valloader: DataLoader,
    config: ExperimentConfig,
    client_epsilon: float | None = None,
    computed_sigma: float | None = None,
    accountant: RDPAccountant | None = None,
    seed: int | None = None,
    mechanism_state: dict | None = None,
) -> FlowerClient:
    if config.privacy.enabled:
        return PerUpdateDPClient(
            model,
            trainloader,
            valloader,
            config,
            client_epsilon=client_epsilon,
            computed_sigma=computed_sigma,
            accountant=accountant,
            seed=seed,
            mechanism_state=mechanism_state,
        )
    return FlowerClient(model, trainloader, valloader, config)
