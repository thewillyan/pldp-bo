from __future__ import annotations

from typing import Any, Callable

from flwr.common import Scalar
from flwr.server.strategy import Strategy

from src.config.loader import ExperimentConfig
from src.server.strategy import FedAvg


def create_strategy(
    config: ExperimentConfig,
    on_round_end: Callable[[int, dict[str, Scalar]], None] | None = None,
) -> Strategy:
    if config.federated.strategy == "fedavg":
        return FedAvg(config).create(on_round_end=on_round_end)
    else:
        raise ValueError(f"Unknown strategy: {config.federated.strategy}")
