from __future__ import annotations

import functools
import logging

import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from torch.utils.data import DataLoader

from src.config.loader import ExperimentConfig, load_config
from src.data import create_validation_loader
from src.device import get_device, to_device
from src.tracking.tracker import ExperimentTracker
from src.models import create_model
from src.server.strategy import MedianRobustAggregation
from src.utils import set_seed

logger = logging.getLogger(__name__)

app = ServerApp()


def _run_global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    config: ExperimentConfig,
    valloader: DataLoader,
    tracker: ExperimentTracker,
) -> MetricRecord:
    model = create_model(config.model, dataset_name=config.data.name)
    model.get_model().load_state_dict(arrays.to_torch_state_dict())
    net = model.get_model().to(get_device())
    net.eval()

    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_images, batch_labels in valloader:
            images, labels = to_device((batch_images, batch_labels))
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = loss / len(valloader)

    metrics: dict[str, float] = {
        "server_loss": avg_loss,
        "accuracy": accuracy,
    }
    tracker.log_round_metrics(server_round, metrics)

    return MetricRecord({"accuracy": accuracy, "loss": avg_loss})


@app.main()
def main(grid: Grid, context: Context) -> None:
    config_path = str(context.run_config.get("config-path", "config/default.yaml"))
    overrides = {
        k: v for k, v in context.run_config.items() if k != "config-path"
    }
    config = load_config(config_path, overrides=overrides)

    set_seed(config.seed, deterministic=config.deterministic)

    valloader = create_validation_loader(config.data)

    tracker = ExperimentTracker(config)
    tracker.start_run()

    global_model = create_model(config.model, dataset_name=config.data.name)
    arrays = ArrayRecord(global_model.get_model().state_dict())

    strategy: FedAvg
    if config.federated.strategy == "pldp_bo":
        strategy = MedianRobustAggregation(
            server_learning_rate=config.federated.server_learning_rate,
            fraction_train=config.federated.fraction_fit,
            fraction_evaluate=config.federated.fraction_evaluate,
            min_train_nodes=config.federated.min_fit_clients,
            min_evaluate_nodes=config.federated.min_evaluate_clients,
            min_available_nodes=config.federated.min_available_nodes,
            tracker=tracker,
        )
    elif config.federated.strategy == "fedprox":
        strategy = FedProx(
            fraction_train=config.federated.fraction_fit,
            fraction_evaluate=config.federated.fraction_evaluate,
            min_train_nodes=config.federated.min_fit_clients,
            min_evaluate_nodes=config.federated.min_evaluate_clients,
            min_available_nodes=config.federated.min_available_nodes,
            proximal_mu=config.federated.proximal_mu,
        )
    else:
        strategy = FedAvg(
            fraction_train=config.federated.fraction_fit,
            fraction_evaluate=config.federated.fraction_evaluate,
            min_train_nodes=config.federated.min_fit_clients,
            min_evaluate_nodes=config.federated.min_evaluate_clients,
            min_available_nodes=config.federated.min_available_nodes,
        )

    if config.federated.server_learning_rate != 1.0 and config.federated.strategy != "pldp_bo":
        logger.warning(
            "server_learning_rate=%.2f is only effective with strategy='pldp_bo' "
            "(current: '%s'). Ignored.",
            config.federated.server_learning_rate,
            config.federated.strategy,
        )

    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=config.federated.num_rounds,
        evaluate_fn=functools.partial(
            _run_global_evaluate,
            config=config,
            valloader=valloader,
            tracker=tracker,
        ),
    )

    tracker.end_run()
