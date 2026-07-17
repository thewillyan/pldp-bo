from __future__ import annotations

import functools
import logging

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
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


def _discover_node_to_partition(
    grid: Grid,
    node_ids: list[int],
) -> dict[int, int]:
    """Discover the mapping from node_id to partition_id via QUERY messages."""
    query_msgs = [
        Message(
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

    node_to_partition: dict[int, int] = {}
    for reply in replies:
        meta = reply.content.config_records.get("config", ConfigRecord())
        node_id = reply.metadata.src_node_id
        p_id = meta.get("partition_id", node_id)
        node_to_partition[node_id] = int(p_id)
    return node_to_partition


def _compute_per_client_budgets(
    grid: Grid,
    config: ExperimentConfig,
) -> tuple[dict[int, float] | None, dict[int, int] | None]:
    """Compute per-client budgets and a mapping from node_id to partition_id.

    Returns (budgets, node_to_partition).
    - Equal division: budgets keyed by node_id, node_to_partition is None
      (_add_budgets_to_messages falls back to node_id as key).
    - Personalized: budgets keyed by partition_id, node_to_partition maps
      node_id -> partition_id.
    """
    total_budget = config.privacy.total_budget
    if total_budget is None:
        return None, None

    node_ids = list(grid.get_node_ids())
    if not node_ids:
        logger.warning("No nodes discovered for budget setup, retrying in 5s...")
        import time
        time.sleep(5)
        node_ids = list(grid.get_node_ids())
    if not node_ids:
        logger.warning("No nodes available for budget setup; skipping per-client budgets")
        return None, None

    # custom strategy: weights from config map
    if config.personalization.enabled and config.personalization.strategy == "custom":
        weight_map = {int(k): v for k, v in config.personalization.client_epsilon_map.items()}
        total_weight = sum(weight_map.values())
        if total_weight <= 0:
            logger.warning("client_epsilon_map sums to 0; falling back to equal division")
            per_client = total_budget / len(node_ids)
            budgets = {nid: per_client for nid in node_ids}
            return budgets, None
        node_to_partition = _discover_node_to_partition(grid, node_ids)
        discovered = set(node_to_partition.values())
        configured = set(weight_map.keys())
        missing = configured - discovered
        extra = discovered - configured
        if missing:
            logger.warning(
                "client_epsilon_map references partition IDs %s not found among %d discovered nodes; "
                "those clients will receive zero budget",
                sorted(missing), len(node_to_partition),
            )
        if extra:
            logger.warning(
                "client_epsilon_map missing entries for partition IDs %s; "
                "those clients will use config-derived defaults",
                sorted(extra),
            )
        budgets = {cid: total_budget * w / total_weight for cid, w in weight_map.items()}
        for nid in node_ids:
            pid = node_to_partition.get(nid, nid)
            if pid not in budgets:
                if nid not in node_to_partition:
                    node_to_partition[nid] = nid
                budgets[pid] = total_budget / len(node_ids)
        return budgets, node_to_partition

    # Other personalization strategies: discover budget weights in a single QUERY round
    if config.personalization.enabled:
        query_msgs = [
            Message(
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

        node_to_partition = {}
        client_weights: dict[int, float] = {}
        for reply in replies:
            meta = reply.content.config_records.get("config", ConfigRecord())
            node_id = reply.metadata.src_node_id
            p_id = meta.get("partition_id", node_id)
            node_to_partition[node_id] = int(p_id)
            weight = meta.get("budget_weight")
            if weight is not None:
                client_weights[int(p_id)] = float(weight)

        if client_weights:
            total_weight = sum(client_weights.values())
            budgets = {
                cid: total_budget * w / total_weight
                for cid, w in client_weights.items()
            }
            for nid in node_ids:
                pid = node_to_partition.get(nid, nid)
                if pid not in budgets:
                    if nid not in node_to_partition:
                        node_to_partition[nid] = nid
                    budgets[pid] = total_budget / len(node_ids)
            return budgets, node_to_partition
        logger.warning(
            "No clients returned budget weights; falling back to equal division",
        )

    # Equal division: key budgets by node_id directly
    per_client = total_budget / len(node_ids)
    budgets = {nid: per_client for nid in node_ids}
    return budgets, None


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

    valloader = create_validation_loader(config.data, seed=config.seed)

    tracker = ExperimentTracker(config)
    tracker.start_run()

    if context.series_id:
        tracker.set_tag("flower_series_id", str(context.series_id))

    global_model = create_model(config.model, dataset_name=config.data.name)
    arrays = ArrayRecord(global_model.get_model().state_dict())

    per_client_budgets, node_to_partition = _compute_per_client_budgets(grid, config)
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
            node_to_partition=node_to_partition,
        )
    elif config.federated.strategy == "fedprox":
        strategy = SafeFedProx(
            fraction_train=config.federated.fraction_fit,
            fraction_evaluate=config.federated.fraction_evaluate,
            min_train_nodes=config.federated.min_fit_clients,
            min_evaluate_nodes=config.federated.min_evaluate_clients,
            min_available_nodes=config.federated.min_available_nodes,
            proximal_mu=config.federated.proximal_mu,
            server_learning_rate=config.federated.server_learning_rate,
            tracker=tracker,
            per_client_budgets=per_client_budgets,
            node_to_partition=node_to_partition,
        )
    else:
        strategy = SafeFedAvg(
            fraction_train=config.federated.fraction_fit,
            fraction_evaluate=config.federated.fraction_evaluate,
            min_train_nodes=config.federated.min_fit_clients,
            min_evaluate_nodes=config.federated.min_evaluate_clients,
            min_available_nodes=config.federated.min_available_nodes,
            server_learning_rate=config.federated.server_learning_rate,
            tracker=tracker,
            per_client_budgets=per_client_budgets,
            node_to_partition=node_to_partition,
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
