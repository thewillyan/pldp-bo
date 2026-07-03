from __future__ import annotations

import mlflow
from mlflow.entities import Run


def get_client() -> mlflow.tracking.MlflowClient:
    return mlflow.tracking.MlflowClient()


def get_run_by_id(run_id: str) -> Run:
    client = get_client()
    try:
        return client.get_run(run_id)
    except mlflow.exceptions.MlflowException as err:
        all_runs = client.search_runs(experiment_ids=["0", "1", "2"])
        for run in all_runs:
            if run.info.run_id.startswith(run_id):
                return run
        raise ValueError(
            f"Run '{run_id}' not found. Use 'list-runs' to see available runs."
        ) from err


def get_run_name(run: Run) -> str:
    name: str = str(run.info.run_name or "")
    if name and not name.startswith("calm-") and not name.startswith("youthful-"):
        return name
    return str(run.info.run_id[:8])


def extract_metrics_by_round(
    run: Run, metric_name: str
) -> tuple[list[int], list[float]]:
    rounds: dict[int, float] = {}
    prefix = "round_"
    suffix = f"_{metric_name}"

    for key, value in run.data.metrics.items():
        if key.startswith(prefix) and key.endswith(suffix):
            parts = key.split("_")
            if len(parts) >= 3:
                try:
                    round_num = int(parts[1])
                    rounds[round_num] = float(value)
                except (ValueError, IndexError):
                    continue

    if not rounds:
        return [], []

    sorted_items = sorted(rounds.items())
    rds, vals = zip(*sorted_items, strict=True)
    return list(rds), list(vals)


def extract_all_round_metrics(
    run: Run,
) -> dict[str, tuple[list[int], list[float]]]:
    metric_names: set[str] = set()
    for key in run.data.metrics:
        if key.startswith("round_") and "_" in key:
            parts = key.split("_")
            if len(parts) >= 3:
                metric_name = "_".join(parts[2:])
                metric_names.add(metric_name)

    result: dict[str, tuple[list[int], list[float]]] = {}
    for name in sorted(metric_names):
        rds, vals = extract_metrics_by_round(run, name)
        if rds:
            result[name] = (rds, vals)
    return result


def list_runs(experiment_name: str | None = None) -> list[Run]:
    client = get_client()
    if experiment_name:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            msg = f"Experiment '{experiment_name}' not found."
            raise ValueError(msg)
        return client.search_runs(
            experiment_ids=[str(experiment.experiment_id)],
        )
    return client.search_runs(experiment_ids=["0", "1", "2"])


def get_run_params(run: Run) -> dict[str, str]:
    return dict(run.data.params)
