from __future__ import annotations

import logging
from typing import Iterable

import mlflow
import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.common import Message
from flwr.serverapp.grid.grid import Grid
from flwr.serverapp.strategy import FedAvg

from src.logging.tracker import ExperimentTracker

log = logging.getLogger(__name__)


class MedianRobustAggregation(FedAvg):
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

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid,
    ) -> Iterable[Message]:
        self._current_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            return None, None

        global_arrays = self._current_arrays
        if global_arrays is None:
            return None, None

        global_keys = list(global_arrays.keys())
        global_ndarrays = [global_arrays[k].numpy() for k in global_keys]

        deltas: list[list[np.ndarray]] = []
        norms: list[float] = []
        reply_contents: list[RecordDict] = []

        for msg in valid_replies:
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
            [1.0 if r <= 1e-12 else min(1.0, median_norm / r) for r in norms],
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

        self._log_client_metrics(server_round, reply_contents)

        metrics = None
        if self.train_metrics_aggr_fn is not None:
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )

        return aggregated, metrics

    def _log_metric(self, key: str, value: float, step: int) -> None:
        if self._tracker is not None:
            self._tracker.log_metrics({key: value}, step=step)
        else:
            mlflow.log_metric(key, value, step=step)

    def _log_metric_stats(self, prefix: str, values: list[float], server_round: int) -> None:
        if not values:
            return
        arr = np.array(values)
        self._log_metric(f"round_{server_round}_{prefix}_mean", float(arr.mean()), step=server_round)
        self._log_metric(f"round_{server_round}_{prefix}_std", float(arr.std()), step=server_round)
        if len(values) >= 3:
            self._log_metric(f"round_{server_round}_{prefix}_min", float(arr.min()), step=server_round)
            self._log_metric(f"round_{server_round}_{prefix}_max", float(arr.max()), step=server_round)
            self._log_metric(f"round_{server_round}_{prefix}_median", float(np.median(arr)), step=server_round)

    def _log_client_metrics(
        self,
        server_round: int,
        reply_contents: list[RecordDict],
    ) -> None:
        epsilons: list[float] = []
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
                self._log_metric(f"round_{server_round}_client_{cid}_epsilon", float(epsilon), step=server_round)

            update_norm = m.get("update_norm")
            if update_norm is not None:
                update_norms.append(float(update_norm))
                self._log_metric(f"round_{server_round}_client_{cid}_update_norm", float(update_norm), step=server_round)

            utility_loss = m.get("utility_loss")
            if utility_loss is not None:
                utility_losses.append(float(utility_loss))
                self._log_metric(f"round_{server_round}_client_{cid}_utility_loss", float(utility_loss), step=server_round)

            cum_eps = m.get("cumulative_epsilon")
            if cum_eps is not None:
                cumulative_epsilons.append(float(cum_eps))
                self._log_metric(f"round_{server_round}_client_{cid}_cumulative_epsilon", float(cum_eps), step=server_round)

        self._log_metric_stats("epsilon", epsilons, server_round)
        self._log_metric_stats("update_norm", update_norms, server_round)
        self._log_metric_stats("utility_loss", utility_losses, server_round)
        self._log_metric_stats("cumulative_epsilon", cumulative_epsilons, server_round)
