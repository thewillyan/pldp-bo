from __future__ import annotations

import argparse
from pathlib import Path

import flwr as fl
import torch

from src.client import create_client
from src.config.loader import load_config
from src.data import create_client_dataloaders
from src.logging.tracker import ExperimentTracker
from src.models import create_model
from src.server import create_strategy


def _set_seed(seed: int) -> None:
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.run_name:
        config.logging.run_name = args.run_name

    _set_seed(config.seed)

    trainloaders, valloader, _ = create_client_dataloaders(config.data, config.seed)

    model = create_model(config.model)

    tracker = ExperimentTracker(config)
    tracker.start_run()

    def on_round_end(round_num: int, metrics: dict) -> None:
        tracker.log_round_metrics(round_num, metrics)

    strategy = create_strategy(config, on_round_end=on_round_end)

    def client_fn(cid: str) -> fl.client.Client:
        client_model = create_model(config.model)
        client = create_client(
            cid=int(cid),
            model=client_model,
            trainloader=trainloaders[int(cid)],
            valloader=valloader,
            config=config,
        )
        return client.to_client()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.data.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.federated.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )

    tracker.end_run()


if __name__ == "__main__":
    main()
