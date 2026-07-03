from __future__ import annotations

from typing import Any, Callable

import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg as FlwrFedAvg
from flwr.common import FitRes, MetricsAggregationFn, Parameters, Scalar


class LoggingFedAvg(FlwrFedAvg):
    def __init__(
        self,
        *args: Any,
        on_round_end: Callable[[int, dict[str, Scalar]], None] | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._on_round_end = on_round_end

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, Exception]],
    ) -> tuple[float | None, dict[str, Scalar]]:
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if self._on_round_end is not None:
            all_metrics: dict[str, Scalar] = {"round": server_round}
            if loss is not None:
                all_metrics["server_loss"] = loss
            all_metrics.update(metrics)
            self._on_round_end(server_round, all_metrics)
        return loss, metrics


class FedAvg:
    def __init__(self, config: Any):
        self.config = config

    def create(
        self, on_round_end: Callable[[int, dict[str, Scalar]], None] | None = None
    ) -> fl.server.strategy.Strategy:
        return LoggingFedAvg(
            fraction_fit=self.config.federated.fraction_fit,
            fraction_evaluate=self.config.federated.fraction_evaluate,
            min_fit_clients=self.config.federated.min_fit_clients,
            min_evaluate_clients=self.config.federated.min_evaluate_clients,
            min_available_clients=self.config.data.num_clients,
            on_round_end=on_round_end,
        )
