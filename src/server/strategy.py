from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.common import Message, Metadata
from flwr.serverapp.grid.grid import Grid
from flwr.serverapp.strategy import FedAvg, FedProx

from src.tracking.tracker import ExperimentTracker

_EPS = 1e-12
_MIN_VALUES_FOR_STATS = 3


def _filter_valid_replies(replies: Iterable[Message]) -> list[Message]:
    """Filter replies to only those with valid training results.

    Replaces Flower's private ``FedAvg._check_and_log_replies`` to avoid
    depending on an unstable internal API.
    """
    valid: list[Message] = []
    for reply in replies:
        metrics_rec = reply.content.metric_records.get("metrics")
        if metrics_rec is None:
            continue
        num_examples = metrics_rec.get("num-examples", 0)
        if num_examples is None or num_examples <= 0:
            continue
        valid.append(reply)
    return valid


def _add_budgets_to_messages(
    messages: Iterable[Message],
    budgets: dict[int, float] | None,
    configrecord_key: str,
    node_to_partition: dict[int, int] | None = None,
) -> Iterable[Message]:
    if budgets is None:
        yield from messages
        return
    for msg in messages:
        dst = msg.metadata.dst_node_id
        partition_id = (node_to_partition or {}).get(dst, dst)
        budget = budgets.get(partition_id)
        if budget is not None:
            content = RecordDict()
            for key, rec in msg.content.config_records.items():
                if key == configrecord_key:
                    content[key] = ConfigRecord({
                        **rec,
                        "per_client_budget": budget,
                    })
                else:
                    content[key] = rec
            for key, rec in msg.content.array_records.items():
                content[key] = rec
            for key, rec in msg.content.metric_records.items():
                content[key] = rec

            orig = msg.metadata
            new_metadata = Metadata(
                run_id=orig.run_id,
                message_id=orig.message_id,
                src_node_id=orig.src_node_id,
                dst_node_id=dst,
                reply_to_message_id=orig.reply_to_message_id,
                group_id=orig.group_id,
                created_at=orig.created_at,
                ttl=orig.ttl,
                message_type=orig.message_type,
                src_task_id=orig.src_task_id,
                dst_task_id=orig.dst_task_id,
            )
            yield Message(content=content, metadata=new_metadata)
        else:
            yield msg


class MetricLoggingMixin:
    _tracker: ExperimentTracker | None
    _per_client_budgets: dict[int, float] | None

    def _log_metric(self, key: str, value: float, step: int) -> None:
        if self._tracker is not None:
            self._tracker.log_metrics({key: value}, step=step)

    def _log_metric_stats(self, prefix: str, values: list[float], server_round: int) -> None:
        if not values:
            return
        arr = np.array(values)
        self._log_metric(f"{prefix}_mean", float(arr.mean()), step=server_round)
        self._log_metric(f"{prefix}_std", float(arr.std()), step=server_round)
        if len(values) >= _MIN_VALUES_FOR_STATS:
            self._log_metric(f"{prefix}_min", float(arr.min()), step=server_round)
            self._log_metric(f"{prefix}_max", float(arr.max()), step=server_round)
            self._log_metric(f"{prefix}_median", float(np.median(arr)), step=server_round)

    def _log_client_metrics(
        self,
        server_round: int,
        reply_contents: list[RecordDict],
    ) -> None:
        epsilons: list[float] = []
        client_epsilons: list[float] = []
        update_norms: list[float] = []
        utility_losses: list[float] = []
        cumulative_epsilons: list[float] = []

        for content in reply_contents:
            m = content.metric_records.get("metrics")
            if m is None:
                continue
            client_id = m.get("client-id")
            if client_id is None:
                continue

            cid = int(client_id)

            epsilon = m.get("epsilon")
            if epsilon is not None:
                epsilons.append(float(epsilon))
                self._log_metric(f"client_{cid}_epsilon", float(epsilon), step=server_round)

            update_norm = m.get("update_norm")
            if update_norm is not None:
                update_norms.append(float(update_norm))
                self._log_metric(f"client_{cid}_update_norm", float(update_norm), step=server_round)

            utility_loss = m.get("utility_loss")
            if utility_loss is not None:
                utility_losses.append(float(utility_loss))
                self._log_metric(f"client_{cid}_utility_loss", float(utility_loss), step=server_round)

            cum_eps_val = m.get("cumulative_epsilon")
            cum_eps = float(cum_eps_val) if cum_eps_val is not None else None
            if cum_eps is not None:
                cumulative_epsilons.append(cum_eps)
                self._log_metric(f"client_{cid}_cumulative_epsilon", cum_eps, step=server_round)

            client_eps = m.get("client_epsilon")
            if client_eps is not None:
                client_epsilons.append(float(client_eps))
                self._log_metric(f"client_{cid}_client_epsilon", float(client_eps), step=server_round)

            if self._per_client_budgets is not None and cum_eps is not None:
                budget = self._per_client_budgets.get(cid)
                if budget is None and self._node_to_partition:
                    reversed_map = {v: k for k, v in self._node_to_partition.items()}
                    budget = self._per_client_budgets.get(reversed_map.get(cid))
                if budget is not None:
                    remaining = max(0.0, budget - float(cum_eps))
                    self._log_metric(f"client_{cid}_remaining_budget", remaining, step=server_round)

        self._log_metric_stats("epsilon", epsilons, server_round)
        self._log_metric_stats("client_epsilon", client_epsilons, server_round)
        self._log_metric_stats("update_norm", update_norms, server_round)
        self._log_metric_stats("utility_loss", utility_losses, server_round)
        self._log_metric_stats("cumulative_epsilon", cumulative_epsilons, server_round)


class MedianRobustAggregation(MetricLoggingMixin, FedAvg):
    def __init__(
        self,
        server_learning_rate: float = 1.0,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn = None,
        evaluate_metrics_aggr_fn = None,
        tracker: ExperimentTracker | None = None,
        per_client_budgets: dict[int, float] | None = None,
        node_to_partition: dict[int, int] | None = None,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )
        self._server_learning_rate = server_learning_rate
        self._current_arrays: ArrayRecord | None = None
        self._tracker = tracker
        self._per_client_budgets = per_client_budgets
        self._node_to_partition = node_to_partition or {}

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid,
    ) -> Iterable[Message]:
        self._current_arrays = arrays
        return _add_budgets_to_messages(
            super().configure_train(server_round, arrays, config, grid),
            self._per_client_budgets,
            self.configrecord_key,
            node_to_partition=self._node_to_partition,
        )

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies = _filter_valid_replies(replies)

        if not valid_replies:
            return None, None

        all_contents = [r.content for r in valid_replies]
        self._log_client_metrics(server_round, all_contents)

        active_replies = [r for r in valid_replies if not _is_budget_exhausted(r)]
        if not active_replies:
            return None, None

        global_arrays = self._current_arrays
        if global_arrays is None:
            return None, None

        global_keys = list(global_arrays.keys())
        global_ndarrays = [global_arrays[k].numpy() for k in global_keys]

        deltas: list[list[np.ndarray]] = []
        norms: list[float] = []
        reply_contents: list[RecordDict] = []

        for msg in active_replies:
            content = msg.content
            client_arrays = content.array_records.get(self.arrayrecord_key)
            if client_arrays is None:
                continue

            client_ndarrays = [client_arrays[k].numpy() for k in global_keys]

            delta = [c - g for c, g in zip(client_ndarrays, global_ndarrays, strict=True)]
            flat = np.concatenate([d.ravel() for d in delta])
            norm = float(np.linalg.norm(flat))

            deltas.append(delta)
            norms.append(norm)
            reply_contents.append(content)

        if not deltas:
            return None, None

        median_norm = float(np.median(norms))
        weights = np.array(
            [1.0 if r <= _EPS else min(1.0, median_norm / r) for r in norms],
            dtype=np.float64,
        )

        total_weight = np.sum(weights)
        if total_weight <= 0:
            return None, None

        norm_weights = weights / total_weight

        num_layers = len(global_ndarrays)
        aggregated_delta = [
            np.sum(
                np.array([nw * d[i] for nw, d in zip(norm_weights, deltas, strict=True)]),
                axis=0,
            )
            for i in range(num_layers)
        ]

        new_ndarrays = [
            g + self._server_learning_rate * ad
            for g, ad in zip(global_ndarrays, aggregated_delta, strict=True)
        ]

        aggregated = ArrayRecord(
            {
                k: Array(np.asarray(v))
                for k, v in zip(global_keys, new_ndarrays, strict=True)
            },
        )

        metrics = None
        if self.train_metrics_aggr_fn is not None:
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )

        return aggregated, metrics


def _is_budget_exhausted(reply: Message) -> bool:
    metrics_rec = reply.content.metric_records.get("metrics")
    if metrics_rec is None:
        return False
    raw = metrics_rec.get("budget_exhausted")
    if raw is None:
        return False
    if isinstance(raw, (list, tuple)):
        return any(bool(v) for v in raw)
    return bool(raw)


class SafeFedAvg(MetricLoggingMixin, FedAvg):
    def __init__(
        self,
        *args,
        tracker: ExperimentTracker | None = None,
        per_client_budgets: dict[int, float] | None = None,
        node_to_partition: dict[int, int] | None = None,
        server_learning_rate: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tracker = tracker
        self._per_client_budgets = per_client_budgets
        self._node_to_partition = node_to_partition or {}
        self._server_learning_rate = server_learning_rate
        self._current_arrays: ArrayRecord | None = None

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid,
    ) -> Iterable[Message]:
        self._current_arrays = arrays
        return _add_budgets_to_messages(
            super().configure_train(server_round, arrays, config, grid),
            self._per_client_budgets,
            self.configrecord_key,
            node_to_partition=self._node_to_partition,
        )

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies = _filter_valid_replies(replies)
        all_contents = [r.content for r in valid_replies]
        self._log_client_metrics(server_round, all_contents)
        active_replies = [r for r in valid_replies if not _is_budget_exhausted(r)]
        if not active_replies:
            return None, None
        result_arrays, metrics = super().aggregate_train(server_round, active_replies)

        if result_arrays is not None and self._current_arrays is not None:
            global_keys = list(self._current_arrays.keys())
            scaled: dict[str, Array] = {}
            for k in global_keys:
                delta = result_arrays[k].numpy() - self._current_arrays[k].numpy()
                scaled[k] = Array(self._current_arrays[k].numpy() + self._server_learning_rate * delta)
            result_arrays = ArrayRecord(scaled)

        return result_arrays, metrics


class SafeFedProx(MetricLoggingMixin, FedProx):
    def __init__(
        self,
        *args,
        tracker: ExperimentTracker | None = None,
        per_client_budgets: dict[int, float] | None = None,
        node_to_partition: dict[int, int] | None = None,
        server_learning_rate: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tracker = tracker
        self._per_client_budgets = per_client_budgets
        self._node_to_partition = node_to_partition or {}
        self._server_learning_rate = server_learning_rate
        self._current_arrays: ArrayRecord | None = None

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid,
    ) -> Iterable[Message]:
        self._current_arrays = arrays
        return _add_budgets_to_messages(
            super().configure_train(server_round, arrays, config, grid),
            self._per_client_budgets,
            self.configrecord_key,
            node_to_partition=self._node_to_partition,
        )

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies = _filter_valid_replies(replies)
        all_contents = [r.content for r in valid_replies]
        self._log_client_metrics(server_round, all_contents)
        active_replies = [r for r in valid_replies if not _is_budget_exhausted(r)]
        if not active_replies:
            return None, None
        result_arrays, metrics = super().aggregate_train(server_round, active_replies)

        if result_arrays is not None and self._current_arrays is not None:
            global_keys = list(self._current_arrays.keys())
            scaled: dict[str, Array] = {}
            for k in global_keys:
                delta = result_arrays[k].numpy() - self._current_arrays[k].numpy()
                scaled[k] = Array(self._current_arrays[k].numpy() + self._server_learning_rate * delta)
            result_arrays = ArrayRecord(scaled)

        return result_arrays, metrics
