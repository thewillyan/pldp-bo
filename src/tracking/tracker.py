from __future__ import annotations

import os
from dataclasses import fields
from typing import Any

import mlflow

from src.config.loader import ExperimentConfig


class ExperimentTracker:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or config.logging.tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config.logging.experiment_name)

    def start_run(self) -> None:
        run_name = self._config.logging.run_name
        mlflow.start_run(run_name=run_name)
        self._log_config()
        if self._config.logging.group:
            mlflow.set_tag("group", self._config.logging.group)

    def end_run(self) -> None:
        mlflow.end_run()

    @staticmethod
    def _flatten_dataclass(obj: object, prefix: str = "") -> dict[str, str]:
        result: dict[str, str] = {}
        for f in fields(obj):
            key = f"{prefix}.{f.name}" if prefix else f.name
            value = getattr(obj, f.name)
            if hasattr(value, "__dataclass_fields__"):
                result.update(ExperimentTracker._flatten_dataclass(value, key))
            else:
                result[key] = str(value)
        return result

    def _log_config(self) -> None:
        config = self._config
        params: dict[str, str] = {}
        params.update(self._flatten_dataclass(config.data, "data"))
        params.update(self._flatten_dataclass(config.model, "model"))
        params.update(self._flatten_dataclass(config.federated, "federated"))
        params.update(self._flatten_dataclass(config.optimizer, "optimizer"))
        params["seed"] = str(config.seed)
        params["deterministic"] = str(config.deterministic)

        if config.privacy.enabled:
            params.update(self._flatten_dataclass(config.privacy, "privacy"))
        if config.personalization.enabled:
            params.update(self._flatten_dataclass(config.personalization, "personalization"))
        if config.bo.enabled:
            params.update(self._flatten_dataclass(config.bo, "bo"))

        mlflow.log_params(params)

    def log_round_metrics(self, round_num: int, metrics: dict[str, Any]) -> None:
        mlflow.log_metrics(metrics, step=round_num)

    def log_final_metrics(self, metrics: dict[str, Any]) -> None:
        prefixed = {f"final_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    @staticmethod
    def set_tag(key: str, value: str) -> None:
        mlflow.set_tag(key, value)

    @staticmethod
    def get_run_id() -> str | None:
        return mlflow.active_run().info.run_id if mlflow.active_run() else None
