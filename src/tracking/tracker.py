from __future__ import annotations

import os
from typing import Any

import mlflow

from src.config.loader import ExperimentConfig


class ExperimentTracker:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        uri = os.environ.get("MLFLOW_TRACKING_URI") or config.logging.tracking_uri
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config.logging.experiment_name)

    def start_run(self) -> None:
        run_name = self._config.logging.run_name
        mlflow.start_run(run_name=run_name)
        self._log_config()

    def end_run(self) -> None:
        mlflow.end_run()

    def _log_config(self) -> None:
        params = {
            "data.name": self._config.data.name,
            "data.num_clients": str(self._config.data.num_clients),
            "data.partition_type": self._config.data.partition_type,
            "data.batch_size": str(self._config.data.batch_size),
            "model.name": self._config.model.name,
            "model.num_classes": str(self._config.model.num_classes),
            "federated.num_rounds": str(self._config.federated.num_rounds),
            "federated.fraction_fit": str(self._config.federated.fraction_fit),
            "federated.local_epochs": str(self._config.federated.local_epochs),
            "federated.strategy": self._config.federated.strategy,
            "optimizer.name": self._config.optimizer.name,
            "optimizer.lr": str(self._config.optimizer.lr),
            "privacy.enabled": str(self._config.privacy.enabled),
            "seed": str(self._config.seed),
        }
        if self._config.privacy.enabled:
            params.update({
                "privacy.mechanism": self._config.privacy.mechanism,
                "privacy.noise_multiplier": str(self._config.privacy.noise_multiplier),
                "privacy.max_grad_norm": str(self._config.privacy.max_grad_norm),
                "privacy.delta": str(self._config.privacy.delta),
                "privacy.accountant": self._config.privacy.accountant,
            })
        if self._config.personalization.enabled:
            params.update({
                "personalization.enabled": "True",
                "personalization.strategy": self._config.personalization.strategy,
                "personalization.epsilon_min": str(self._config.personalization.epsilon_min),
                "personalization.epsilon_max": str(self._config.personalization.epsilon_max),
                "personalization.epsilon_base": str(self._config.personalization.epsilon_base),
                "personalization.track_cumulative": str(self._config.personalization.track_cumulative),
            })
        if self._config.bo.enabled:
            params.update({
                "bo.enabled": "True",
                "bo.warmup_rounds": str(self._config.bo.warmup_rounds),
                "bo.epsilon_min": str(self._config.bo.epsilon_min),
                "bo.epsilon_max": str(self._config.bo.epsilon_max),
                "bo.epsilon_budget": str(self._config.bo.epsilon_budget),
                "bo.optimization_metric": self._config.bo.optimization_metric,
                "bo.grid_points": str(self._config.bo.grid_points),
                "bo.acquisition_penalty": str(self._config.bo.acquisition_penalty),
                "bo.gp_kernel": self._config.bo.gp_kernel,
                "bo.observation_noise": str(self._config.bo.observation_noise),
            })
        if self._config.federated.server_learning_rate != 1.0:
            params["federated.server_learning_rate"] = str(self._config.federated.server_learning_rate)
        params["deterministic"] = str(self._config.deterministic)
        mlflow.log_params(params)

    def log_round_metrics(self, round_num: int, metrics: dict[str, Any]) -> None:
        prefixed = {f"round_{round_num}_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed, step=round_num)

    def log_final_metrics(self, metrics: dict[str, Any]) -> None:
        prefixed = {f"final_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    @staticmethod
    def get_run_id() -> str | None:
        return mlflow.active_run().info.run_id if mlflow.active_run() else None
