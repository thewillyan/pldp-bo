from __future__ import annotations

import hashlib
import json
from unittest.mock import patch

import pytest

from src.config.loader import ExperimentConfig
from src.privacy.bo_scheduler import WARMUP_GRID
from src.tracking.tracker import (
    ExperimentTracker,
    data_hash,
    dataset_sizes,
    experiment_name,
    git_hash,
    run_name,
)


class TestExperimentTracker:
    @staticmethod
    def _make_tracker() -> ExperimentTracker:
        config = ExperimentConfig()
        with (
            patch("src.tracking.tracker.mlflow.set_tracking_uri"),
            patch("src.tracking.tracker.mlflow.set_experiment"),
        ):
            return ExperimentTracker(config)

    def test_start_run_calls_mlflow(self) -> None:
        config = ExperimentConfig()
        config.method = "nonprivate"
        with (
            patch("src.tracking.tracker.mlflow.set_tracking_uri"),
            patch("src.tracking.tracker.mlflow.set_experiment"),
        ):
            tracker = ExperimentTracker(config)

        with patch("src.tracking.tracker.mlflow") as mock_mlflow:
            tracker.start_run()
            mock_mlflow.start_run.assert_called_once_with(
                run_name=f"nonprivate_seed{config.seed}",
            )
            mock_mlflow.log_params.assert_called_once()

    def test_end_run_calls_mlflow(self) -> None:
        tracker = self._make_tracker()

        with patch("src.tracking.tracker.mlflow") as mock_mlflow:
            tracker.end_run()
            mock_mlflow.end_run.assert_called_once()

    def test_log_round_metrics(self) -> None:
        tracker = self._make_tracker()

        with patch("src.tracking.tracker.mlflow") as mock_mlflow:
            tracker.log_round_metrics(5, {"loss": 0.5, "accuracy": 0.9})
            mock_mlflow.log_metrics.assert_called_once_with({"loss": 0.5, "accuracy": 0.9}, step=5)

    def test_log_metrics_with_step(self) -> None:
        tracker = self._make_tracker()

        with patch("src.tracking.tracker.mlflow") as mock_mlflow:
            tracker.log_metrics({"loss": 0.4}, step=3)
            mock_mlflow.log_metrics.assert_called_once_with({"loss": 0.4}, step=3)

    def test_log_metrics_without_step(self) -> None:
        tracker = self._make_tracker()

        with patch("src.tracking.tracker.mlflow") as mock_mlflow:
            tracker.log_metrics({"loss": 0.4})
            mock_mlflow.log_metrics.assert_called_once_with({"loss": 0.4}, step=None)

    def test_log_artifact(self) -> None:
        tracker = self._make_tracker()

        with patch("src.tracking.tracker.mlflow") as mock_mlflow:
            tracker.log_artifact("/path/to/file")
            mock_mlflow.log_artifact.assert_called_once_with("/path/to/file")

    def test_get_run_id_returns_none_when_no_active_run(self) -> None:
        self._make_tracker()

        with patch("src.tracking.tracker.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            result = ExperimentTracker.get_run_id()
            assert result is None


class TestNamingDerivation:
    def _make_config(self) -> ExperimentConfig:
        config = ExperimentConfig()
        config.method = "pldpbo_snr"
        config.seed = 3
        return config

    def test_experiment_name_iid(self) -> None:
        config = self._make_config()
        config.data.name = "mnist"
        config.data.partition_type = "iid"
        assert experiment_name(config) == "mnist_iid"

    def test_experiment_name_dirichlet_alpha(self) -> None:
        config = self._make_config()
        config.data.name = "mnist"
        config.data.partition_type = "dirichlet"
        config.data.partition_alpha = 0.5
        assert experiment_name(config) == "mnist_dirichlet_0.5"

    def test_experiment_name_noniid_uses_alpha_0_5(self) -> None:
        config = self._make_config()
        config.data.name = "cifar100"
        config.data.partition_type = "noniid"
        assert experiment_name(config) == "cifar100_dirichlet_0.5"

    def test_experiment_name_pathological(self) -> None:
        config = self._make_config()
        config.data.name = "cifar100"
        config.data.partition_type = "pathological"
        assert experiment_name(config) == "cifar100_pathological"

    def test_experiment_name_writer_is_natural(self) -> None:
        config = self._make_config()
        config.data.name = "femnist"
        config.data.partition_type = "writer"
        assert experiment_name(config) == "femnist_natural"

    def test_run_name_method_and_seed(self) -> None:
        config = self._make_config()
        assert run_name(config) == "pldpbo_snr_seed3"


class TestTrackerTags:
    def test_start_run_uses_derived_names_and_sets_spec_tags(self) -> None:
        config = ExperimentConfig()
        config.method = "pldpbo_snr"
        config.seed = 3
        config.data.name = "mnist"
        config.data.partition_type = "dirichlet"
        config.data.partition_alpha = 0.5

        with (
            patch("src.tracking.tracker.mlflow.set_tracking_uri"),
            patch("src.tracking.tracker.mlflow.set_experiment") as mock_set_experiment,
            patch("src.tracking.tracker.subprocess.run") as mock_run,
        ):
            tracker = ExperimentTracker(config)
            mock_run.return_value.stdout = "abc123def\n"
            with patch("src.tracking.tracker.mlflow") as mock_mlflow:
                tracker.start_run()
        mock_set_experiment.assert_called_once_with("mnist_dirichlet_0.5")
        mock_mlflow.start_run.assert_called_once_with(run_name="pldpbo_snr_seed3")
        tags = mock_mlflow.set_tags.call_args.args[0]
        assert tags["dataset"] == "mnist"
        assert tags["partition"] == "dirichlet_0.5"
        assert tags["method"] == "pldpbo_snr"
        assert tags["seed"] == "3"
        assert tags["code_git_hash"] == "abc123def"
        assert len(tags["config_version"]) == 64
        int(tags["config_version"], 16)

    def test_config_version_is_stable(self) -> None:
        from src.config.locked import config_version as locked_version
        config = ExperimentConfig()
        assert run_name(config)  # sanity: config loads
        assert locked_version() == locked_version()
        assert len(locked_version()) == 64

    def test_git_hash_fallback_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_args: object, **_kwargs: object) -> None:
            raise FileNotFoundError("not a git repo")

        monkeypatch.setattr("src.tracking.tracker.subprocess.run", _boom)
        assert git_hash() == "unknown"


class TestDataHash:
    def test_digest_over_sorted_files_with_relpaths(
        self, tmp_path: pytest.TempPathFactory,
    ) -> None:
        root = tmp_path / "MNIST"
        root.mkdir()
        (root / "b.txt").write_bytes(b"xyz")
        (root / "a.bin").write_bytes(b"abc")
        config = ExperimentConfig()
        config.data.name = "mnist"
        config.data.data_dir = str(tmp_path)
        expected = hashlib.sha256(
            b"a.bin\x00abc\x00b.txt\x00xyz\x00",
        ).hexdigest()
        assert data_hash(config) == expected

    def test_content_change_changes_digest(self, tmp_path: pytest.TempPathFactory) -> None:
        root = tmp_path / "MNIST"
        root.mkdir()
        (root / "a.bin").write_bytes(b"abc")
        config = ExperimentConfig()
        config.data.name = "mnist"
        config.data.data_dir = str(tmp_path)
        before = data_hash(config)
        (root / "a.bin").write_bytes(b"zzz")
        assert data_hash(config) != before

    def test_missing_dir_returns_none(self, tmp_path: pytest.TempPathFactory) -> None:
        config = ExperimentConfig()
        config.data.name = "mnist"
        config.data.data_dir = str(tmp_path)
        assert data_hash(config) is None


class TestDatasetSizes:
    def test_mnist_constants(self) -> None:
        config = ExperimentConfig()
        config.data.name = "mnist"
        assert dataset_sizes(config) == {"train": 60000, "test": 10000, "writers": None}

    def test_cifar10_constants(self) -> None:
        config = ExperimentConfig()
        config.data.name = "cifar10"
        assert dataset_sizes(config) == {"train": 50000, "test": 10000, "writers": None}

    def test_femnist_reads_processed_files(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        processed = tmp_path / "FEMNIST" / "processed"
        processed.mkdir(parents=True)
        for name in ("femnist_train.pt", "femnist_test.pt", "femnist_user_keys.pt"):
            (processed / name).write_bytes(b"x")
        monkeypatch.setattr(
            "src.tracking.tracker.femnist_counts",
            lambda _root: (10, 5, 3),
        )
        config = ExperimentConfig()
        config.data.name = "femnist"
        config.data.data_dir = str(tmp_path)
        assert dataset_sizes(config) == {"train": 10, "test": 5, "writers": 3}

    def test_femnist_missing_files_none(
        self, tmp_path: pytest.TempPathFactory,
    ) -> None:
        config = ExperimentConfig()
        config.data.name = "femnist"
        config.data.data_dir = str(tmp_path)
        assert dataset_sizes(config) is None

    def test_unknown_dataset_none(self) -> None:
        config = ExperimentConfig()
        config.data.name = "unknown"
        assert dataset_sizes(config) is None


class TestSpecParams:
    SECTION42_KEYS = {
        "T", "K", "rho", "E", "B", "eta_server", "local_opt",
        "clip_norm", "alpha0", "B_RDP", "R_min", "R_max",
        "warmup_points", "warmup_sum_nominal", "lambda_aq", "kernel", "G",
        "N", "mu_fedprox", "model", "dataset_sizes", "partition_kwargs",
        "seeds", "validation_frac", "aggregation", "enforce_budget",
        "dataset_root", "data_hash",
    }

    def _make_bo_config(self) -> ExperimentConfig:
        config = ExperimentConfig()
        config.method = "pldpbo_snr"
        config.seed = 3
        config.data.name = "mnist"
        config.data.partition_type = "dirichlet"
        config.data.partition_alpha = 0.5
        config.data.batch_size = 64
        config.privacy.enabled = True
        config.privacy.accountant_mode = "rdp_native"
        config.privacy.total_budget = 10.0
        config.privacy.enforce_budget = True
        config.bo.enabled = True
        return config

    def _params_for(self, config: ExperimentConfig, tmp_path: pytest.TempPathFactory) -> dict:
        root = tmp_path / "MNIST"
        root.mkdir()
        (root / "a.bin").write_bytes(b"abc")
        config.data.data_dir = str(tmp_path)
        with (
            patch("src.tracking.tracker.mlflow.set_tracking_uri"),
            patch("src.tracking.tracker.mlflow.set_experiment"),
            patch("src.tracking.tracker.subprocess.run") as mock_run,
        ):
            mock_run.return_value.stdout = "abc123def\n"
            tracker = ExperimentTracker(config)
            with patch("src.tracking.tracker.mlflow") as mock_mlflow:
                tracker.start_run()
        params: dict[str, str] = mock_mlflow.log_params.call_args.args[0]
        return params

    def test_all_section42_params_logged_for_bo_method(
        self, tmp_path: pytest.TempPathFactory,
    ) -> None:
        config = self._make_bo_config()
        params = self._params_for(config, tmp_path)
        assert set(params) >= self.SECTION42_KEYS

    def test_section42_values(self, tmp_path: pytest.TempPathFactory) -> None:
        config = self._make_bo_config()
        config.data.num_clients = 100
        config.federated.num_rounds = 200
        config.federated.fraction_fit = 0.1
        config.federated.local_epochs = 5
        config.federated.server_learning_rate = 0.01
        config.federated.aggregation = "attenuation"
        config.optimizer.name = "sgd"
        config.optimizer.momentum = 0.9
        config.bo.grid_points = 50
        config.bo.rdp_min = 0.01
        config.bo.rdp_max = 2.0
        config.bo.acquisition_penalty = 0.1
        params = self._params_for(config, tmp_path)
        assert params["T"] == "200"
        assert params["K"] == "100"
        assert params["rho"] == "0.1"
        assert params["E"] == "5"
        assert params["B"] == "64"
        assert params["eta_server"] == "0.01"
        assert params["local_opt"] == "sgd_momentum0.9"
        assert params["clip_norm"] == "1.0"
        assert params["alpha0"] == "10.0"
        assert params["B_RDP"] == "10.0"
        assert params["R_min"] == "0.01"
        assert params["R_max"] == "2.0"
        assert params["warmup_points"] == json.dumps(list(WARMUP_GRID))
        assert params["warmup_sum_nominal"] == str(sum(WARMUP_GRID))
        assert params["lambda_aq"] == "0.1"
        assert params["kernel"] == "matern52"
        assert params["G"] == "50"
        assert params["N"] == "3"
        assert params["mu_fedprox"] == "0.0"
        assert params["model"] == "cnn"
        assert json.loads(params["dataset_sizes"]) == {
            "train": 60000, "test": 10000, "writers": None,
        }
        assert json.loads(params["partition_kwargs"]) == {
            "type": "dirichlet", "alpha": 0.5,
        }
        assert json.loads(params["seeds"]) == {"global": 3, "numpy": 3, "torch": 3}
        assert params["validation_frac"] == "0.1"
        assert params["aggregation"] == "attenuation"
        assert params["enforce_budget"] == "true"
        assert params["dataset_root"] == str(tmp_path)
        assert len(params["data_hash"]) == 64

    def test_privacy_params_absent_for_nonprivate(
        self, tmp_path: pytest.TempPathFactory,
    ) -> None:
        config = self._make_bo_config()
        config.method = "nonprivate"
        config.privacy.enabled = False
        config.bo.enabled = False
        config.federated.aggregation = "plain"
        params = self._params_for(config, tmp_path)
        assert "clip_norm" not in params
        assert "alpha0" not in params
        assert "B_RDP" not in params
        assert "R_min" not in params
        assert "warmup_points" not in params
        assert "lambda_aq" not in params
        assert "kernel" not in params
        assert "G" not in params
        assert "enforce_budget" not in params
        assert "data_hash" in params
        assert "dataset_sizes" in params

    def test_legacy_params_still_logged(self, tmp_path: pytest.TempPathFactory) -> None:
        config = self._make_bo_config()
        params = self._params_for(config, tmp_path)
        assert "data.name" in params
        assert "federated.num_rounds" in params
        assert "privacy.accountant_mode" in params
