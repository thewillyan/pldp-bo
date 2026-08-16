from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Any

import mlflow

from src.config.loader import ExperimentConfig
from src.config.locked import config_version as locked_config_version
from src.data.femnist import femnist_counts
from src.data.partitioner import build_partition_kwargs
from src.privacy.bo_scheduler import WARMUP_GRID, WARMUP_SUM_NOMINAL

# Dataset directory under ``data_dir`` per dataset name (§4.2 data_hash /
# dataset_sizes). MNIST and CIFAR use the torchvision layout; FEMNIST uses
# the LEAF extraction layout (spec §9.9).
_DATASET_DIRS: dict[str, str] = {
    "mnist": "MNIST",
    "cifar10": "cifar-10-batches-py",
    "cifar100": "cifar-100-python",
    "femnist": "FEMNIST/processed",
}

_DATASET_SIZES: dict[str, tuple[int, int, None]] = {
    "mnist": (60000, 10000, None),
    "cifar10": (50000, 10000, None),
    "cifar100": (50000, 10000, None),
}


def partition_label(config: ExperimentConfig) -> str:
    """§4.1 partition label: iid / dirichlet_<alpha> / pathological / natural."""
    pt = config.data.partition_type
    if pt == "noniid":
        return "dirichlet_0.5"
    if pt == "dirichlet":
        return f"dirichlet_{config.data.partition_alpha}"
    if pt == "writer":
        return "natural"
    return pt


def experiment_name(config: ExperimentConfig) -> str:
    """§4.1 experiment name ``<dataset>_<partition>``."""
    return f"{config.data.name.lower()}_{partition_label(config)}"


def run_name(config: ExperimentConfig) -> str:
    """§4.1 run name ``<method>_seed<NN>``."""
    return f"{config.method}_seed{config.seed}"


def git_hash() -> str:
    """HEAD commit hash for the code_git_hash tag; 'unknown' outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except subprocess.SubprocessError, OSError:
        return "unknown"
    return result.stdout.strip()


def data_hash(config: ExperimentConfig) -> str | None:
    """sha256 over the sorted dataset files (§4.2 data_hash); None when absent.

    The digest covers each file's path relative to the dataset directory and
    its bytes, so renaming a file also changes the digest.
    """
    rel = _DATASET_DIRS.get(config.data.name.lower())
    if rel is None:
        return None
    root = Path(config.data.data_dir) / rel
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def dataset_sizes(config: ExperimentConfig) -> dict[str, int | None] | None:
    """§4.2 dataset_sizes JSON payload (train/test/writers); None when unknown."""
    name = config.data.name.lower()
    if name in _DATASET_SIZES:
        n_train, n_test, n_writers = _DATASET_SIZES[name]
        return {"train": n_train, "test": n_test, "writers": n_writers}
    if name == "femnist":
        processed = Path(config.data.data_dir) / "FEMNIST" / "processed"
        files = (
            "femnist_train.pt",
            "femnist_test.pt",
            "femnist_user_keys.pt",
        )
        if all((processed / f).is_file() for f in files):
            n_train, n_test, n_writers = femnist_counts(config.data.data_dir)
            return {"train": n_train, "test": n_test, "writers": n_writers}
        return None
    return None


class ExperimentTracker:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self.experiment_name = experiment_name(config)
        self.run_name = run_name(config)
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or config.logging.tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self) -> None:
        mlflow.start_run(run_name=self.run_name)
        self._log_config()
        self._log_tags()

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

        params.update(self._spec_params(config))

        mlflow.log_params(params)

    def _spec_params(self, config: ExperimentConfig) -> dict[str, str]:
        """§4.2 params under the spec names (EXPERIMENTS-TODO.md §4.2)."""
        data, fed, opt, priv, bo = (
            config.data,
            config.federated,
            config.optimizer,
            config.privacy,
            config.bo,
        )
        params: dict[str, str] = {
            "T": str(fed.num_rounds),
            "K": str(data.num_clients),
            "rho": str(fed.fraction_fit),
            "E": str(fed.local_epochs),
            "B": str(data.batch_size),
            "eta_server": str(fed.server_learning_rate),
            "local_opt": f"{opt.name}_momentum{opt.momentum}",
            "N": str(config.seed),
            "mu_fedprox": str(fed.proximal_mu),
            "model": config.model.name,
            "validation_frac": str(data.val_split),
            "aggregation": fed.aggregation,
            "dataset_root": data.data_dir,
        }
        sizes = dataset_sizes(config)
        if sizes is not None:
            params["dataset_sizes"] = json.dumps(sizes)
        params["partition_kwargs"] = json.dumps(
            build_partition_kwargs(data.partition_type, data.partition_alpha),
        )
        params["seeds"] = json.dumps(
            {
                "global": config.seed,
                "numpy": config.seed,
                "torch": config.seed,
            }
        )
        digest = data_hash(config)
        if digest is not None:
            params["data_hash"] = digest

        if priv.enabled:
            params["clip_norm"] = str(priv.update_clip_norm)
            params["alpha0"] = str(priv.rdp_alpha)
            params["enforce_budget"] = str(priv.enforce_budget).lower()
            if priv.total_budget is not None:
                params["B_RDP"] = str(priv.total_budget)
            params["warmup_points"] = json.dumps(list(WARMUP_GRID))
            params["warmup_sum_nominal"] = str(WARMUP_SUM_NOMINAL)
        if bo.enabled:
            params["R_min"] = str(bo.rdp_min)
            params["R_max"] = str(bo.rdp_max)
            params["lambda_aq"] = str(bo.acquisition_penalty)
            params["kernel"] = bo.gp_kernel
            params["G"] = str(bo.grid_points)
        return params

    def _log_tags(self) -> None:
        config = self._config
        tags: dict[str, str] = {
            "dataset": config.data.name.lower(),
            "partition": partition_label(config),
            "method": config.method,
            "seed": str(config.seed),
            "config_version": locked_config_version(),
            "code_git_hash": git_hash(),
        }
        if config.logging.group:
            tags["group"] = config.logging.group
        mlflow.set_tags(tags)

    def log_round_metrics(self, round_num: int, metrics: dict[str, Any]) -> None:
        mlflow.log_metrics(metrics, step=round_num)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    @staticmethod
    def set_tag(key: str, value: str) -> None:
        mlflow.set_tag(key, value)

    @staticmethod
    def get_run_id() -> str | None:
        run = mlflow.active_run()
        return run.info.run_id if run else None
