from __future__ import annotations

import functools
import logging

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx  # noqa: F401  # used in type annotation
from torch.utils.data import DataLoader

from src.config.loader import ExperimentConfig, load_config
from src.data import create_validation_loader
from src.device import get_device, to_device
from src.models import create_model
from src.server.strategy import MedianRobustAggregation, SafeFedAvg, SafeFedProx
from src.tracking.tracker import ExperimentTracker
from src.utils import set_seed

logger = logging.getLogger(__name__)

app = ServerApp()


def _compute_per_client_budgets(
    grid: Grid,
    config: ExperimentConfig,
) -> dict[int, float] | None:
    total_budget = config.privacy.total_budget
    if total_budget is None:
        return None

    num_clients = max(config.data.num_clients, 1)

    # custom strategy: proportional from config map, no setup needed
    if config.personalization.enabled and config.personalization.strategy == "custom":
        eps_map = {int(k): v for k, v in config.personalization.client_epsilon_map.items()}
        total_weight = sum(eps_map.values())
        if total_weight <= 0:
            logger.warning("client_epsilon_map sums to 0; falling back to equal division")
            per_client = total_budget / num_clients
            return {cid: per_client for cid in range(num_clients)}
        return {cid: total_budget * eps / total_weight for cid, eps in eps_map.items()}

    # Other personalization strategies: discover metadata via QUERY
    if config.personalization.enabled:
        node_ids = list(grid.get_node_ids())
        if node_ids:
            query_msgs = [
                grid.create_message(
                    content=RecordDict({
                        "config": ConfigRecord({"task": "personalization_metadata"}),
                    }),
                    message_type="query",
                    dst_node_id=nid,
                    group_id="pldp-budget-setup",
                )
                for nid in node_ids
            ]
            replies = list(grid.send_and_receive(query_msgs, timeout=30.0))

            client_epsilons: dict[int, float] = {}
            for reply in replies:
                meta = reply.content.config_records.get("config", ConfigRecord())
                cid = reply.metadata.src_node_id
                eps = meta.get("personalization_epsilon")
                if eps is not None:
                    client_epsilons[cid] = float(eps)

            if client_epsilons:
                total_weight = sum(client_epsilons.values())
                return {cid: total_budget * eps / total_weight for cid, eps in client_epsilons.items()}
            logger.warning(
                "No clients returned personalization metadata; falling back to equal division",
            )
        else:
            logger.warning("No nodes available for setup; falling back to equal division")

    # Equal division
    per_client = total_budget / num_clients
    return {cid: per_client for cid in range(num_clients)}


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

    per_client_budgets = _compute_per_client_budgets(grid, config)
    if per_client_budgets is not None:
        logger.info(
            "Computed per-client budgets from total_budget=%.2f across %d clients",
            config.privacy.total_budget,
            len(per_client_budgets),
        )

    strategy: FedAvg | FedProx | MedianRobustAggregation
    if config.federated.strategy == "pldp_bo":
        strategy = MedianRobustAggregation(
            server_learning_rate=config.federated.server_learning_rate,
            fraction_train=config.federated.fraction_fit,
            fraction_evaluate=config.federated.fraction_evaluate,
            min_train_nodes=config.federated.min_fit_clients,
            min_evaluate_nodes=config.federated.min_evaluate_clients,
            min_available_nodes=config.federated.min_available_nodes,
            tracker=tracker,
            per_client_budgets=per_client_budgets,
        )
    elif config.federated.strategy == "fedprox":
        strategy = SafeFedProx(
            fraction_train=config.federated.fraction_fit,
            fraction_evaluate=config.federated.fraction_evaluate,
            min_train_nodes=config.federated.min_fit_clients,
            min_evaluate_nodes=config.federated.min_evaluate_clients,
            min_available_nodes=config.federated.min_available_nodes,
            proximal_mu=config.federated.proximal_mu,
            per_client_budgets=per_client_budgets,
        )
    else:
        strategy = SafeFedAvg(
            fraction_train=config.federated.fraction_fit,
            fraction_evaluate=config.federated.fraction_evaluate,
            min_train_nodes=config.federated.min_fit_clients,
            min_evaluate_nodes=config.federated.min_evaluate_clients,
            min_available_nodes=config.federated.min_available_nodes,
            per_client_budgets=per_client_budgets,
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
