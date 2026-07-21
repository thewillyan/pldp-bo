from __future__ import annotations

import re
import warnings

import mlflow
from mlflow.entities import Run


def get_client() -> mlflow.tracking.MlflowClient:
    return mlflow.tracking.MlflowClient()


def get_run_by_id(run_id: str) -> Run:
    client = get_client()
    try:
        return client.get_run(run_id)
    except mlflow.exceptions.MlflowException as err:
        experiment_ids = [exp.experiment_id for exp in client.search_experiments()]
        all_runs = client.search_runs(experiment_ids=experiment_ids)
        matches = [run for run in all_runs if run.info.run_id.startswith(run_id)]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple runs match prefix '{run_id}': "
                f"{[m.info.run_id for m in matches]}. Use full run ID.",
            )
        if matches:
            return matches[0]
        raise ValueError(
            f"Run '{run_id}' not found. Use 'list-runs' to see available runs.",
        ) from err


def get_run_name(run: Run) -> str:
    name: str = str(run.info.run_name or "")
    if name and not _is_mlflow_auto_name(name):
        return name
    return str(run.info.run_id[:8])


# MLflow auto-generated names match "adjective-noun-####" (three hyphen-separated parts,
# third is a positive integer). This avoids false positives on user names that use
# underscores (common in this project's configs: "pldp_bo_mnist_iid_nun").
_AUTO_NAME_RE = re.compile(r"^[a-z]+-[a-z]+-\d+$")
_per_client_key_re = re.compile(r"(?:^|_)client_\d+_")

def _is_mlflow_auto_name(name: str) -> bool:
    return bool(_AUTO_NAME_RE.match(name))


def _get_metric_history(run_id: str, metric_name: str) -> list[tuple[int, float]]:
    client = get_client()
    try:
        metrics = client.get_metric_history(run_id, metric_name)
    except Exception as e:
        warnings.warn(
            f"Failed to fetch metric history for '{metric_name}' "
            f"(run {run_id[:8]}): {e}",
            stacklevel=2,
        )
        return []
    return [(m.step, m.value) for m in metrics]


def _dedup_by_step(pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    seen: dict[int, float] = {}
    for step, value in pairs:
        seen[step] = value
    return sorted(seen.items())


def extract_metrics_by_round(
    run: Run, metric_name: str,
) -> tuple[list[int], list[float]]:
    history = _get_metric_history(run.info.run_id, metric_name)
    if history:
        deduped = _dedup_by_step(history)
        rds, vals = zip(*deduped, strict=True) if deduped else ([], [])
        return list(rds), list(vals)

    rounds: dict[int, float] = {}
    prefix = "round_"
    suffix = f"_{metric_name}"

    for key, value in run.data.metrics.items():
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        if _per_client_key_re.search(key):
            continue
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
    run_id = run.info.run_id
    result: dict[str, tuple[list[int], list[float]]] = {}

    for key in run.data.metrics:
        if (
            key.startswith("round_")
            or key.endswith("_mean")
            or key.endswith("_std")
            or _per_client_key_re.search(key)
        ):
            continue
        history = _get_metric_history(run_id, key)
        if history:
            deduped = _dedup_by_step(history)
            if deduped:
                rds, vals = zip(*deduped, strict=True)
                result[key] = (list(rds), list(vals))

    if result:
        return result

    pairs: dict[str, dict[int, float]] = {}
    _round_metric_re = re.compile(r"^round_(\d+)_(?!client_\d+_)(.+)$")
    for key, value in run.data.metrics.items():
        m = _round_metric_re.match(key)
        if m:
            round_num = int(m.group(1))
            metric_name = m.group(2)
            pairs.setdefault(metric_name, {})[round_num] = float(value)

    for name in sorted(pairs):
        sorted_items = sorted(pairs[name].items())
        rds, vals = zip(*sorted_items, strict=True)
        result[name] = (list(rds), list(vals))
    return result


def extract_per_client_metric(
    run: Run,
    client_id: int,
    metric_name: str,
) -> tuple[list[int], list[float]]:
    history = _get_metric_history(run.info.run_id, f"client_{client_id}_{metric_name}")
    if history:
        deduped = _dedup_by_step(history)
        rds, vals = zip(*deduped, strict=True) if deduped else ([], [])
        return list(rds), list(vals)

    rounds: dict[int, float] = {}
    prefix = "round_"
    pattern = f"_client_{client_id}_{metric_name}"

    for key, value in run.data.metrics.items():
        if key.startswith(prefix) and key.endswith(pattern):
            parts = key.split("_")
            if len(parts) >= 5:
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


def extract_round_stats(
    run: Run,
    stat_name: str,
    aggs: tuple[str, ...] = ("mean", "std", "min", "max", "median"),
) -> tuple[list[int], dict[str, list[float]]]:
    run_id = run.info.run_id
    result: dict[str, dict[int, float]] = {agg: {} for agg in aggs}
    has_new_format = False

    for agg in aggs:
        key = f"{stat_name}_{agg}"
        history = _get_metric_history(run_id, key)
        if history:
            has_new_format = True
            for step, value in history:
                result[agg][step] = value

    if has_new_format:
        all_rounds: set[int] = set()
        for agg_data in result.values():
            all_rounds.update(agg_data.keys())
        if not all_rounds:
            return [], {}
        sorted_rounds = sorted(all_rounds)
        output: dict[str, list[float]] = {}
        for agg in aggs:
            output[agg] = [result[agg].get(r, float("nan")) for r in sorted_rounds]
        return sorted_rounds, output

    for key, value in run.data.metrics.items():
        if not key.startswith("round_"):
            continue
        parts = key.split("_")
        if len(parts) < 4:
            continue
        agg = parts[-1]
        if agg not in aggs:
            continue
        actual_stat = "_".join(parts[2:-1])
        if actual_stat != stat_name:
            continue
        try:
            round_num = int(parts[1])
            result[agg][round_num] = float(value)
        except (ValueError, IndexError):
            continue

    all_rounds = set()
    for agg_data in result.values():
        all_rounds.update(agg_data.keys())

    if not all_rounds:
        return [], {}

    sorted_rounds = sorted(all_rounds)
    output = {}
    for agg in aggs:
        output[agg] = [result[agg].get(r, float("nan")) for r in sorted_rounds]

    return sorted_rounds, output


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
    experiments = client.search_experiments()
    return client.search_runs(
        experiment_ids=[str(e.experiment_id) for e in experiments],
    )


def get_run_params(run: Run) -> dict[str, str]:
    return dict(run.data.params)
