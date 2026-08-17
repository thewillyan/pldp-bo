from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Sequence
from time import perf_counter

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from torch import nn

from src.config.loader import ExperimentConfig, load_config
from src.config.locked import assert_locked_config
from src.data import create_test_loader
from src.device import get_device, to_device
from src.models import create_model
from src.models.base import BaseModel
from src.server.strategy import (
    MedianRobustAggregation,
    MetricLoggingMixin,
    SafeFedAvg,
)
from src.tracking.tracker import ExperimentTracker
from src.utils import set_seed

logger = logging.getLogger(__name__)

app = ServerApp()


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Macro-averaged F1 over classes present in ``y_true``.

    Per-class precision/recall with zero denominators counted as 0; classes
    never predicted nor correct get F1 = 0 (sklearn-style ``zero_division``).
    """
    classes = sorted(set(y_true))
    if not classes:
        return 0.0
    f1_scores: list[float] = []
    for cls in classes:
        tp = fp = fn = 0
        for t, p in zip(y_true, y_pred, strict=True):
            if p == cls:
                if t == cls:
                    tp += 1
                else:
                    fp += 1
            elif t == cls:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall > 0:
            f1_scores.append(2 * precision * recall / (precision + recall))
        else:
            f1_scores.append(0.0)
    return sum(f1_scores) / len(f1_scores)


def _run_global_test_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    config: ExperimentConfig,
    model: BaseModel,
    tracker: ExperimentTracker,
) -> MetricRecord | None:
    """Evaluate the global model on the official test set (IMPL-08).

    Logs ``acc_test`` (top-1) and ``f1_test`` (macro-F1) at ``step=round``
    for rounds >= 1; returns the MetricRecord (round 0 is skipped so the
    untrained model never pollutes the curves).
    """
    if server_round == 0:
        return None

    model.set_weights(arrays.to_numpy_ndarrays())
    net = model.get_model().to(get_device())
    net.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch_images, batch_labels in create_test_loader(config.data):
            images, labels = to_device((batch_images, batch_labels))
            outputs = net(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += int((predicted == labels).sum().item())
            y_true.extend(int(v) for v in labels.tolist())
            y_pred.extend(int(v) for v in predicted.tolist())

    if total == 0:
        return None
    acc_test = correct / total
    f1_test = _macro_f1(y_true, y_pred)
    tracker.log_metrics({"acc_test": acc_test, "f1_test": f1_test}, step=server_round)
    return MetricRecord({"acc_test": acc_test, "f1_test": f1_test})


def _run_femnist_client_test_accuracy(
    grid: Grid,
    final_arrays: ArrayRecord,
    tracker: ExperimentTracker,
) -> None:
    """Ask each client to evaluate the final model on its writers' test samples.

    FEMNIST only (spec §9.7): each client evaluates the final global model on
    the test samples belonging to its own writers and reports the mean; the
    server writes a deterministic ``client_test_acc.json`` artifact keyed by
    partition_id.
    """
    node_ids = list(grid.get_node_ids())
    if not node_ids:
        logger.warning("No nodes available for per-client test accuracy; skipping")
        return

    query_msgs = [
        Message(
            content=RecordDict(
                {
                    "config": ConfigRecord({"task": "client_test_accuracy"}),
                    "arrays": ArrayRecord(dict(final_arrays.items())),
                }
            ),
            message_type="query",
            dst_node_id=nid,
            group_id="pldp-client-test-accuracy",
        )
        for nid in node_ids
    ]
    replies = list(grid.send_and_receive(query_msgs, timeout=300.0))

    per_client: dict[int, float] = {}
    for reply in replies:
        meta = reply.content.config_records.get("config", ConfigRecord())
        pid_raw = meta.get("partition_id")
        acc_raw = meta.get("test_accuracy")
        if pid_raw is None or acc_raw is None:
            logger.warning(
                "Per-client test accuracy reply missing partition_id/accuracy: %s",
                meta,
            )
            continue
        if not isinstance(pid_raw, (int, float)) or not isinstance(acc_raw, (int, float)):
            logger.warning(
                "Per-client test accuracy reply has invalid values: %s",
                meta,
            )
            continue
        per_client[int(pid_raw)] = float(acc_raw)

    if not per_client:
        logger.warning("No per-client test accuracy replies received")
        return

    payload = {
        "client_test_acc": {str(pid): acc for pid, acc in sorted(per_client.items())},
    }
    fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="client_test_acc_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, sort_keys=True, indent=2)
        tracker.log_artifact(tmp_path)
    finally:
        os.unlink(tmp_path)
    logger.info(
        "Logged client_test_acc.json for %d clients",
        len(per_client),
    )


def _write_client_state_artifact(
    strategy: MetricLoggingMixin,
    config: ExperimentConfig,
    tracker: ExperimentTracker,
    wall_time: float,
    num_rounds: int,
) -> None:
    """Final client_state.json artifact + §4.3 final metrics (IMPL-11 §4.4).

    Privacy disabled: no privacy fields are logged, the artifact is skipped
    (§4.4 N/A rules). Fixed baselines accumulate naturally from client
    replies (candidate == final, phase ``bo``, zero enforcement).
    """
    if not config.privacy.enabled:
        return
    state = strategy.get_client_state()
    if not state:
        logger.warning("No per-client state accumulated; skipping client_state.json")
        return

    payload = {
        "client_state": {str(cid): s for cid, s in sorted(state.items())},
    }
    fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="client_state_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, sort_keys=True, indent=2)
        # mlflow.log_artifact keeps the file's basename, so the temp name
        # would reach the store; scripts/verify reads the canonical
        # ``client_state.json`` (IMPL-14 Task 6).
        artifact_path = os.path.join(os.path.dirname(tmp_path), "client_state.json")
        os.replace(tmp_path, artifact_path)
        try:
            tracker.log_artifact(artifact_path)
        finally:
            os.unlink(artifact_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    logger.info("Logged client_state.json for %d clients", len(state))

    final_rdps = [s["cum_rdp"][-1] for s in state.values() if s["cum_rdp"]]
    budget = config.privacy.total_budget
    if final_rdps and budget:
        tracker.log_metrics(
            {"budget_utilization": float(np.mean(final_rdps)) / float(budget)},
            step=num_rounds,
        )
    if wall_time > 0:
        tracker.log_metrics(
            {
                "bo_overhead_pct": 100.0 * strategy.get_bo_time_total() / wall_time,
            },
            step=num_rounds,
        )


def _make_strategy(
    config: ExperimentConfig,
    tracker: ExperimentTracker | None,
    per_client_budgets: dict[int, float] | None,
    node_to_partition: dict[int, int] | None,
) -> SafeFedAvg | MedianRobustAggregation:
    """Select the server strategy from federated.aggregation (IMPL-09 §9.5).

    attenuation (all private methods, including the fixed baselines) →
    MedianRobustAggregation; plain (nonprivate) → SafeFedAvg. The proximal
    term is client-side (IMPL-04), so fedprox_fixed never uses SafeFedProx.
    """
    if config.federated.aggregation == "plain":
        return SafeFedAvg(
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
    return MedianRobustAggregation(
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


def _compute_per_client_budgets(
    grid: Grid,
    config: ExperimentConfig,
) -> tuple[dict[int, float] | None, dict[int, int] | None]:
    """Compute per-client budgets keyed by node_id.

    Returns (budgets, node_to_partition). IMPL-09 §9.5: every client gets the
    full B_RDP as its lifetime budget (flat, not divided by K), so
    node_to_partition is always None and _add_budgets_to_messages falls back
    to node_id as key.
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

    budgets = {nid: total_budget for nid in node_ids}
    return budgets, None


@app.main()
def main(grid: Grid, context: Context) -> None:
    config_path = str(context.run_config.get("config-path", "config/default.yaml"))

    app_overrides = json.loads(str(context.run_config.get("app_config_overrides", "{}")))
    overrides = {
        k: v
        for k, v in context.run_config.items()
        if k not in ("config-path", "app_config_overrides")
    }
    overrides.update(app_overrides)
    config = load_config(config_path, overrides=overrides)

    assert_locked_config(config)

    set_seed(config.seed, deterministic=config.deterministic)

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

    strategy = _make_strategy(config, tracker, per_client_budgets, node_to_partition)

    last_arrays = arrays

    def global_test_evaluate(
        server_round: int,
        round_arrays: ArrayRecord,
    ) -> MetricRecord | None:
        nonlocal last_arrays
        last_arrays = round_arrays
        return _run_global_test_evaluate(
            server_round,
            round_arrays,
            config,
            global_model,
            tracker,
        )

    wall_start = perf_counter()
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=config.federated.num_rounds,
        evaluate_fn=global_test_evaluate,
    )
    wall_time = perf_counter() - wall_start

    _write_client_state_artifact(
        strategy,
        config,
        tracker,
        wall_time=wall_time,
        num_rounds=config.federated.num_rounds,
    )

    if config.data.name == "femnist":
        _run_femnist_client_test_accuracy(grid, last_arrays, tracker)

    tracker.end_run()
