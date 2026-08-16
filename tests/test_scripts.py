from __future__ import annotations

import json
import os
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlflow
import pytest
import yaml

from src.config.loader import load_config
from src.config.locked import collect_violations
from src.config.locked import config_version as locked_config_version


def _load_script(path: str, module_name: str) -> types.ModuleType:
    import importlib.machinery
    import sys

    loader = importlib.machinery.SourceFileLoader(module_name, path)
    mod = types.ModuleType(module_name)
    mod.__file__ = path
    prev = sys.modules.get(module_name)
    sys.modules[module_name] = mod
    try:
        loader.exec_module(mod)
    finally:
        if prev is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = prev
    return mod


_run = _load_script("scripts/run", "_run_script")
_plot = _load_script("scripts/plot", "_plot_script")
_gen = _load_script("scripts/gen_matrix_configs", "_gen_script")
_verify = _load_script("scripts/verify", "_verify_script")


def _make_run(
    tracking_uri: str,
    experiment: str,
    run_name: str,
    status: str = "FINISHED",
    tag: str | None = None,
) -> str:
    """Create a run directly in *tracking_uri* (real sqlite DB, no mocking)."""
    prev = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment)
        exp_id = exp.experiment_id if exp else client.create_experiment(experiment)
        run = client.create_run(exp_id, run_name=run_name)
        run_id = run.info.run_id
        if tag is not None:
            client.set_tag(run_id, "config_version", tag)
        client.set_terminated(run_id, status=status)
        return run_id
    finally:
        mlflow.set_tracking_uri(prev)


def _make_verify_run(
    tracking_uri: str,
    experiment: str,
    run_name: str,
    *,
    method: str,
    dataset: str,
    partition: str,
    seed: int,
    status: str = "FINISHED",
    tag: str | None = None,
    params: dict[str, str] | None = None,
    state: dict | None = None,
) -> str:
    """Create a §4-schema run; optionally log a client_state.json artifact.

    Callers must chdir into a tmp dir first (artifact root resolves from CWD).
    """
    prev = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment)
        exp_id = exp.experiment_id if exp else client.create_experiment(experiment)
        run_id = client.create_run(exp_id, run_name=run_name).info.run_id
        for key, value in {
            "dataset": dataset,
            "partition": partition,
            "method": method,
            "seed": str(seed),
            "config_version": tag or locked_config_version(),
        }.items():
            client.set_tag(run_id, key, value)
        for key, value in (params or {}).items():
            client.log_param(run_id, key, value)
        if state is not None:
            tmp = os.path.join(tempfile.mkdtemp(), "client_state.json")
            with open(tmp, "w") as f:
                json.dump(state, f)
            try:
                client.log_artifact(run_id, tmp, artifact_path="")
            finally:
                os.unlink(tmp)
        client.set_terminated(run_id, status=status)
        return run_id
    finally:
        mlflow.set_tracking_uri(prev)


class TestNeedsQuoting:
    def test_numeric_string(self) -> None:
        assert _run._needs_quoting("42") is False

    def test_float_string(self) -> None:
        assert _run._needs_quoting("3.14") is False

    def test_negative_numeric(self) -> None:
        assert _run._needs_quoting("-5") is False

    def test_text_string(self) -> None:
        assert _run._needs_quoting("foo") is True

    def test_empty_string(self) -> None:
        assert _run._needs_quoting("") is True


class TestParseJsonFromStream:
    def test_single_line_object(self) -> None:
        data, depth = _run._parse_json_from_stream('{"a": 1}\n', [], 0)
        assert data == {"a": 1}
        assert depth == 0

    def test_multiline_object(self) -> None:
        buffer: list[str] = []
        _, depth = _run._parse_json_from_stream("{\n", buffer, 0)
        assert depth == 1
        data, depth = _run._parse_json_from_stream('  "a": 1\n', buffer, 1)
        assert depth == 1
        data, depth = _run._parse_json_from_stream("}\n", buffer, 1)
        assert data == {"a": 1}
        assert depth == 0

    def test_partial_json_returns_none(self) -> None:
        data, depth = _run._parse_json_from_stream('{"incomplete": ', [], 0)
        assert data is None

    def test_nested_json(self) -> None:
        buffer: list[str] = []
        _, depth = _run._parse_json_from_stream('{"outer": {"inner": 1}}\n', buffer, 0)
        # depth goes 0 → 1 → 2 → 1 → 0 in a single line
        assert depth == 0

    def test_non_dict_json_skipped(self) -> None:
        data, depth = _run._parse_json_from_stream('"just a string"\n', [], 0)
        assert data is None


class TestJsonSaveDir:
    def test_sqlite_uri(self) -> None:
        result = _run._json_save_dir("sqlite:///mlruns/foo/mlflow.db")
        assert result == Path("mlruns/foo/json")

    def test_sqlite_uri_nested(self) -> None:
        result = _run._json_save_dir("sqlite:///mlruns/group/my_group/mlflow.db")
        assert result == Path("mlruns/group/my_group/json")

    def test_default_uri(self) -> None:
        result = _run._json_save_dir("mlflow://localhost:5000")
        assert result == Path("json")


class TestResolveRunId:
    def test_returns_run_id_when_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "abc123"
        mock_client.search_runs.return_value = [mock_run]
        monkeypatch.setattr(
            _run.mlflow.tracking,
            "MlflowClient",
            lambda: mock_client,
        )
        run_id = _run._resolve_run_id("sqlite:///test.db", "my-run")
        assert run_id == "abc123"
        _, kwargs = mock_client.search_runs.call_args
        assert "my-run" in kwargs["filter_string"]

    def test_scoped_to_experiment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_exp = MagicMock()
        mock_exp.experiment_id = "42"
        mock_client.get_experiment_by_name.return_value = mock_exp
        mock_client.search_runs.return_value = []
        monkeypatch.setattr(
            _run.mlflow.tracking,
            "MlflowClient",
            lambda: mock_client,
        )
        _run._resolve_run_id("sqlite:///test.db", "my-run", experiment="mnist_iid")
        mock_client.get_experiment_by_name.assert_called_once_with("mnist_iid")
        _, kwargs = mock_client.search_runs.call_args
        assert kwargs["experiment_ids"] == ["42"]

    def test_scoped_missing_experiment_returns_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None
        monkeypatch.setattr(
            _run.mlflow.tracking,
            "MlflowClient",
            lambda: mock_client,
        )
        run_id = _run._resolve_run_id("sqlite:///test.db", "my-run", experiment="nope")
        assert run_id is None

    def test_returns_none_when_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client.search_runs.return_value = []
        monkeypatch.setattr(
            _run.mlflow.tracking,
            "MlflowClient",
            lambda: mock_client,
        )
        run_id = _run._resolve_run_id("sqlite:///test.db", "missing-run")
        assert run_id is None

    def test_escapes_run_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client.search_runs.return_value = []
        monkeypatch.setattr(
            _run.mlflow.tracking,
            "MlflowClient",
            lambda: mock_client,
        )
        _run._resolve_run_id("sqlite:///test.db", "test's run")
        _, kwargs = mock_client.search_runs.call_args
        assert "test's run" not in kwargs["filter_string"]
        assert "test" in kwargs["filter_string"]

    def test_escapes_backslash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client.search_runs.return_value = []
        monkeypatch.setattr(
            _run.mlflow.tracking,
            "MlflowClient",
            lambda: mock_client,
        )
        _run._resolve_run_id("sqlite:///test.db", "test\\name")
        _, kwargs = mock_client.search_runs.call_args
        assert "test\\\\name" in kwargs["filter_string"]


class TestFormatTime:
    def test_no_end_time(self) -> None:
        result = _plot._format_time(1000, None)
        assert "→" in result
        assert "—" in result

    def test_with_elapsed_seconds(self) -> None:
        result = _plot._format_time(0, 1000)
        assert "→" in result
        assert "1s" in result

    def test_with_elapsed_minutes(self) -> None:
        result = _plot._format_time(0, 125_000)
        assert "2m" in result
        assert "5s" in result

    def test_with_elapsed_hours(self) -> None:
        start = 1_000_000
        end = start + 3 * 3600 * 1000 + 15 * 60 * 1000
        result = _plot._format_time(start, end)
        assert "3h" in result
        assert "15m" in result

    def test_with_elapsed_days(self) -> None:
        start = 0
        end = 2 * 86400 * 1000 + 3600 * 1000
        result = _plot._format_time(start, end)
        assert "2d" in result
        assert "1h" in result


class TestSqliteUriRe:
    def test_matches_simple_path(self) -> None:
        m = _run._SQLITE_URI_RE.match("sqlite:///mlruns/foo.db")
        assert m is not None
        assert m.group(1) == "mlruns/foo.db"

    def test_matches_absolute_path(self) -> None:
        m = _run._SQLITE_URI_RE.match("sqlite:////abs/path/to/db")
        assert m is not None
        assert m.group(1) == "/abs/path/to/db"

    def test_no_match_for_http(self) -> None:
        assert _run._SQLITE_URI_RE.match("http://localhost:5000") is None


class TestGenMatrixConfigs:
    @pytest.fixture(autouse=True)
    def _out_dir(self, tmp_path: Path) -> None:
        self.out_dir = tmp_path / "matrix"

    def _write(self) -> list[Path]:
        return _gen.write_configs(self.out_dir)

    def test_emits_all_100_cells(self) -> None:
        files = self._write()
        assert len(files) == 100
        names = {f.name for f in files}
        expected = {
            f"{dataset}_{part}_{method}.yaml"
            for dataset, parts in _gen.CELLS.items()
            for part in parts
            for method in _gen.METHODS
        }
        assert names == expected

    def test_cell_count_invariants(self) -> None:
        assert len(_gen.CELLS) == 3
        assert sum(len(parts) for parts in _gen.CELLS.values()) == 10
        assert _gen.CELLS["femnist"] == ["natural"]
        assert _gen.CELLS["mnist"] == [
            "iid",
            "dirichlet_1.0",
            "dirichlet_0.5",
            "dirichlet_0.1",
            "pathological",
        ]
        assert _gen.CELLS["cifar100"] == [
            "iid",
            "dirichlet_0.5",
            "dirichlet_0.1",
            "pathological",
        ]
        assert len(_gen.METHODS) == 10

    def test_every_config_passes_locked_assertion(self) -> None:
        for path in self._write():
            cfg = load_config(str(path))
            assert collect_violations(cfg) == [], path.name

    def test_locked_constants_in_file(self) -> None:
        paths = {p.name: p for p in self._write()}
        cfg = load_config(str(paths["mnist_dirichlet_0.5_pldpbo_snr.yaml"]))
        assert cfg.assert_locked_config is True
        assert cfg.seed == 0
        assert cfg.method == "pldpbo_snr"
        fed, data, opt, priv, bo = cfg.federated, cfg.data, cfg.optimizer, cfg.privacy, cfg.bo
        assert fed.num_rounds == 200
        assert data.num_clients == 100
        assert fed.fraction_fit == 0.1
        assert fed.min_fit_clients == 10
        assert fed.local_epochs == 5
        assert data.batch_size == 64
        assert opt.lr == 0.01
        assert opt.momentum == 0.9
        assert opt.weight_decay == 0.0
        assert opt.gradient_clip_norm == 0.0
        assert priv.update_clip_norm == 1.0
        assert priv.rdp_alpha == 10.0
        assert priv.total_budget == 10.0
        assert priv.enforce_budget is True
        assert bo.rdp_min == 0.01
        assert bo.rdp_max == 2.0
        assert bo.acquisition_penalty == 0.1
        assert bo.grid_points == 50
        assert bo.gp_kernel == "matern52"
        assert bo.min_warmup == 10
        assert bo.bounds_strategy == "global"
        assert data.val_split == 0.1
        assert cfg.personalization.enabled is False
        assert fed.min_available_nodes == 100

    def test_method_specific_fields(self) -> None:
        by_name = {p.name: load_config(str(p)) for p in self._write()}
        np = by_name["mnist_iid_nonprivate.yaml"]
        assert np.privacy.enabled is False
        assert np.bo.enabled is False
        assert np.federated.aggregation == "plain"
        assert np.federated.proximal_mu == 0.0

        fixed = by_name["mnist_iid_dpfedavg_fixed.yaml"]
        assert fixed.privacy.enabled is True
        assert fixed.bo.enabled is False
        assert fixed.privacy.fixed_rdp_target == 0.5
        assert fixed.federated.aggregation == "attenuation"

        fedprox = by_name["mnist_iid_fedprox_fixed.yaml"]
        assert fedprox.federated.proximal_mu == 0.01
        assert fedprox.bo.enabled is False
        assert fedprox.federated.aggregation == "attenuation"

        metrics = {
            "pldpbo_nun": "nun",
            "pldpbo_utility": "utility",
            "pldpbo_retention": "utility_retention",
            "pldpbo_efficiency": "utility_efficiency",
            "pldpbo_perremaining": "utility_per_remaining",
            "pldpbo_snr": "snr",
            "pldpbo_agreement": "logit_disagreement",
        }
        for method, metric in metrics.items():
            cfg = by_name[f"mnist_iid_{method}.yaml"]
            assert cfg.privacy.enabled is True
            assert cfg.bo.enabled is True
            assert cfg.bo.optimization_metric == metric
            assert cfg.federated.aggregation == "attenuation"

    def test_dataset_partition_specific_fields(self) -> None:
        by_name = {p.name: load_config(str(p)) for p in self._write()}
        mnist = by_name["mnist_iid_pldpbo_nun.yaml"]
        assert mnist.model.name == "mlp"
        assert mnist.model.num_classes == 10
        assert mnist.data.partition_type == "iid"
        assert mnist.data.partition_min_samples == 30

        c100 = by_name["cifar100_iid_pldpbo_nun.yaml"]
        assert c100.model.name == "cnn"
        assert c100.model.num_classes == 100

        d01 = by_name["mnist_dirichlet_0.1_pldpbo_nun.yaml"]
        assert d01.data.partition_type == "dirichlet"
        assert d01.data.partition_alpha == 0.1

        patho = by_name["cifar100_pathological_pldpbo_nun.yaml"]
        assert patho.data.partition_type == "pathological"

        femnist = by_name["femnist_natural_pldpbo_nun.yaml"]
        assert femnist.data.partition_type == "writer"
        assert femnist.model.name == "cnn"
        assert femnist.model.num_classes == 62

    def test_deterministic_output(self) -> None:
        first = self._write()
        second = self._write()
        assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]

    def test_config_version_metadata(self) -> None:
        paths = {p.name: p for p in self._write()}
        raw = yaml.safe_load(paths["femnist_natural_pldpbo_agreement.yaml"].read_text())
        assert raw["config_version"] == locked_config_version()


class TestParseSeeds:
    def test_range(self) -> None:
        assert _run._parse_seeds("0-11") == list(range(12))

    def test_single(self) -> None:
        assert _run._parse_seeds("3") == [3]

    def test_partial_range(self) -> None:
        assert _run._parse_seeds("2-4") == [2, 3, 4]


class TestMatrixInventory:
    @pytest.fixture(autouse=True)
    def _cells_dir(self, tmp_path: Path) -> None:
        self.cells_dir = tmp_path / "cells"
        _gen.write_configs(self.cells_dir)

    @pytest.fixture
    def mlflow_uri(self, tmp_path: Path) -> str:
        return f"sqlite:///{tmp_path}/mlflow.db"

    def test_all_missing_on_empty_db(self, mlflow_uri: str) -> None:
        inv = _run.matrix_inventory(mlflow_uri, self.cells_dir, [0, 1])
        assert len(inv) == 200
        assert all(r.status == "missing" for r in inv)
        assert all(r.run_id is None for r in inv)

    def test_done_when_finished_and_matching_tag(self, mlflow_uri: str) -> None:
        run_id = _make_run(
            mlflow_uri, "mnist_iid", "pldpbo_snr_seed0", tag=locked_config_version(),
        )
        by_cell = {
            (r.cell, r.run_name): r
            for r in _run.matrix_inventory(mlflow_uri, self.cells_dir, [0, 1])
        }
        done = by_cell[("mnist_iid", "pldpbo_snr_seed0")]
        assert done.status == "done"
        assert done.run_id == run_id
        assert by_cell[("mnist_iid", "pldpbo_snr_seed1")].status == "missing"

    def test_failed_when_crashed(self, mlflow_uri: str) -> None:
        _make_run(mlflow_uri, "mnist_iid", "pldpbo_snr_seed0", status="FAILED")
        by_cell = {
            (r.cell, r.run_name): r
            for r in _run.matrix_inventory(mlflow_uri, self.cells_dir, [0])
        }
        assert by_cell[("mnist_iid", "pldpbo_snr_seed0")].status == "failed"

    def test_failed_when_stale_config_version(self, mlflow_uri: str) -> None:
        _make_run(mlflow_uri, "mnist_iid", "pldpbo_snr_seed0", tag="deadbeef")
        by_cell = {
            (r.cell, r.run_name): r
            for r in _run.matrix_inventory(mlflow_uri, self.cells_dir, [0])
        }
        assert by_cell[("mnist_iid", "pldpbo_snr_seed0")].status == "failed"

    def test_experiment_scoped(self, mlflow_uri: str) -> None:
        # A FINISHED matching run under a different experiment must not count.
        _make_run(
            mlflow_uri, "cifar100_iid", "pldpbo_snr_seed0", tag=locked_config_version(),
        )
        by_cell = {
            (r.cell, r.run_name): r
            for r in _run.matrix_inventory(mlflow_uri, self.cells_dir, [0])
        }
        assert by_cell[("mnist_iid", "pldpbo_snr_seed0")].status == "missing"
        assert by_cell[("cifar100_iid", "pldpbo_snr_seed0")].status == "done"

    def test_plan_excludes_done(self, mlflow_uri: str) -> None:
        _make_run(
            mlflow_uri, "mnist_iid", "pldpbo_snr_seed0", tag=locked_config_version(),
        )
        plan = _run.matrix_plan(_run.matrix_inventory(mlflow_uri, self.cells_dir, [0, 1]))
        assert len(plan) == 199
        assert ("mnist_iid", "pldpbo_snr_seed0") not in {
            (r.cell, r.run_name) for r in plan
        }

    def test_dry_run_reports_1200_missing(
        self, mlflow_uri: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setattr(_run, "CONFIG_DIR", self.cells_dir)
        args = SimpleNamespace(
            dry_run=True,
            dataset=[],
            partition=[],
            method=[],
            seeds="0-11",
            tracking_uri=mlflow_uri,
            num_clients=None,
        )
        _run.cmd_matrix(args)
        out = capsys.readouterr().out
        assert "missing: 1200" in out
        assert "done: 0" in out


class TestRunMatrix:
    @pytest.fixture(autouse=True)
    def _cells_dir(self, tmp_path: Path) -> None:
        self.cells_dir = tmp_path / "cells"
        _gen.write_configs(self.cells_dir)

    @pytest.fixture
    def mlflow_uri(self, tmp_path: Path) -> str:
        return f"sqlite:///{tmp_path}/mlflow.db"

    @pytest.fixture
    def cell_config(self) -> Path:
        return next(
            p for p in self.cells_dir.glob("mnist_iid_pldpbo_snr.yaml")
        )

    def test_deletes_failed_run_and_relaunches(
        self, mlflow_uri: str, cell_config: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        old_id = _make_run(mlflow_uri, "mnist_iid", "pldpbo_snr_seed0", status="FAILED")
        calls: list[tuple[str, int, list[str]]] = []

        def fake_cmd_single(
            config_path: str, num_clients: int, overrides: list[str],
            experiment: str | None = None,
        ) -> tuple[str, str] | None:
            calls.append((config_path, num_clients, overrides))
            assert experiment == "mnist_iid"

            assert config_path == str(cell_config)
            assert num_clients == 100
            assert overrides == ["seed=0"]
            _make_run(
                mlflow_uri, "mnist_iid", "pldpbo_snr_seed0", tag=locked_config_version(),
            )
            return None

        monkeypatch.setattr(_run, "cmd_single", fake_cmd_single)
        inventory = _run.matrix_inventory(
            mlflow_uri, self.cells_dir, [0],
            datasets=["mnist"], partitions=["iid"], methods=["pldpbo_snr"],
        )
        _run.run_matrix(mlflow_uri, _run.matrix_plan(inventory))

        assert len(calls) == 1
        prev = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(mlflow_uri)
        try:
            client = mlflow.tracking.MlflowClient()
            old = client.get_run(old_id)
            assert old.info.lifecycle_stage == "deleted"
            by_cell = {
                (r.cell, r.run_name): r
                for r in _run.matrix_inventory(mlflow_uri, self.cells_dir, [0])
            }
            done = by_cell[("mnist_iid", "pldpbo_snr_seed0")]
            assert done.status == "done"
            assert done.run_id != old_id
        finally:
            mlflow.set_tracking_uri(prev)

    def test_skips_done_runs(self, mlflow_uri: str, monkeypatch: pytest.MonkeyPatch) -> None:
        _make_run(
            mlflow_uri, "mnist_iid", "pldpbo_snr_seed0", tag=locked_config_version(),
        )
        calls: list[tuple[str, int, list[str]]] = []

        def fake_cmd_single(
            config_path: str, num_clients: int, overrides: list[str],
            experiment: str | None = None,
        ) -> tuple[str, str] | None:
            calls.append((config_path, num_clients, overrides))
            assert experiment == "mnist_iid"

            return None

        monkeypatch.setattr(_run, "cmd_single", fake_cmd_single)
        inventory = _run.matrix_inventory(
            mlflow_uri, self.cells_dir, [0],
            datasets=["mnist"], partitions=["iid"], methods=["pldpbo_snr"],
        )
        _run.run_matrix(mlflow_uri, _run.matrix_plan(inventory))
        assert calls == []

    def test_num_clients_read_from_config(
        self, mlflow_uri: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[str, int, list[str]]] = []

        def fake_cmd_single(
            config_path: str, num_clients: int, overrides: list[str],
            experiment: str | None = None,
        ) -> tuple[str, str] | None:
            calls.append((config_path, num_clients, overrides))
            assert experiment == "mnist_iid"

            return None

        monkeypatch.setattr(_run, "cmd_single", fake_cmd_single)
        inventory = _run.matrix_inventory(
            mlflow_uri, self.cells_dir, [0],
            datasets=["mnist"], partitions=["iid"], methods=["pldpbo_snr"],
        )
        _run.run_matrix(mlflow_uri, _run.matrix_plan(inventory))
        assert calls and calls[0][1] == 100

    def test_num_clients_override_wins(
        self, mlflow_uri: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[str, int, list[str]]] = []

        def fake_cmd_single(
            config_path: str, num_clients: int, overrides: list[str],
            experiment: str | None = None,
        ) -> tuple[str, str] | None:
            calls.append((config_path, num_clients, overrides))
            assert experiment == "mnist_iid"

            return None

        monkeypatch.setattr(_run, "cmd_single", fake_cmd_single)
        inventory = _run.matrix_inventory(
            mlflow_uri, self.cells_dir, [0],
            datasets=["mnist"], partitions=["iid"], methods=["pldpbo_snr"],
        )
        _run.run_matrix(mlflow_uri, _run.matrix_plan(inventory), num_clients=20)
        assert calls and calls[0][1] == 20


class TestVerifyDiscovery:
    @pytest.fixture(autouse=True)
    def _tmp_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        self.mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"

    def test_includes_only_finished_current_version(self) -> None:
        good = _make_verify_run(
            self.mlflow_uri, "mnist_iid", "pldpbo_snr_seed0",
            method="pldpbo_snr", dataset="mnist", partition="iid", seed=0,
        )
        _make_verify_run(
            self.mlflow_uri, "mnist_iid", "pldpbo_snr_seed1",
            method="pldpbo_snr", dataset="mnist", partition="iid", seed=1,
            status="FAILED",
        )
        _make_verify_run(
            self.mlflow_uri, "mnist_iid", "pldpbo_snr_seed2",
            method="pldpbo_snr", dataset="mnist", partition="iid", seed=2,
            tag="deadbeef",
        )
        runs = _verify._discover_runs(self.mlflow_uri)
        assert [r.run_id for r in runs] == [good]

    def test_spans_experiments(self) -> None:
        _make_verify_run(
            self.mlflow_uri, "mnist_iid", "pldpbo_snr_seed0",
            method="pldpbo_snr", dataset="mnist", partition="iid", seed=0,
        )
        _make_verify_run(
            self.mlflow_uri, "cifar100_dirichlet_0.1", "pldpbo_nun_seed3",
            method="pldpbo_nun", dataset="cifar100", partition="dirichlet_0.1", seed=3,
        )
        runs = _verify._discover_runs(self.mlflow_uri)
        assert sorted(r.experiment for r in runs) == [
            "cifar100_dirichlet_0.1", "mnist_iid",
        ]

    def test_filters_by_tags(self) -> None:
        _make_verify_run(
            self.mlflow_uri, "mnist_iid", "pldpbo_snr_seed0",
            method="pldpbo_snr", dataset="mnist", partition="iid", seed=0,
        )
        _make_verify_run(
            self.mlflow_uri, "mnist_iid", "pldpbo_snr_seed7",
            method="pldpbo_snr", dataset="mnist", partition="iid", seed=7,
        )
        _make_verify_run(
            self.mlflow_uri, "cifar100_iid", "pldpbo_snr_seed0",
            method="pldpbo_snr", dataset="cifar100", partition="iid", seed=0,
        )
        runs = _verify._discover_runs(
            self.mlflow_uri,
            datasets=["mnist"], partitions=["iid"],
            methods=["pldpbo_snr"], seeds=[0],
        )
        assert len(runs) == 1
        assert runs[0].seed == 0
        assert runs[0].experiment == "mnist_iid"

    def test_carries_params(self) -> None:
        _make_verify_run(
            self.mlflow_uri, "mnist_iid", "pldpbo_snr_seed0",
            method="pldpbo_snr", dataset="mnist", partition="iid", seed=0,
            params={"T": "200", "K": "100", "B_RDP": "10.0", "dataset_root": "/data"},
        )
        runs = _verify._discover_runs(self.mlflow_uri)
        assert runs[0].params["T"] == "200"
        assert runs[0].params["B_RDP"] == "10.0"

    def test_loads_client_state_artifact(self) -> None:
        state = {"0": {"acct_cost": [0.01, 0.02]}}
        run_id = _make_verify_run(
            self.mlflow_uri, "mnist_iid", "pldpbo_snr_seed0",
            method="pldpbo_snr", dataset="mnist", partition="iid", seed=0,
            state=state,
        )
        runs = _verify._discover_runs(self.mlflow_uri)
        assert runs[0].run_id == run_id
        assert runs[0].client_state == state

    def test_no_artifact_gives_none(self) -> None:
        _make_verify_run(
            self.mlflow_uri, "mnist_iid", "pldpbo_snr_seed0",
            method="pldpbo_snr", dataset="mnist", partition="iid", seed=0,
        )
        runs = _verify._discover_runs(self.mlflow_uri)
        assert runs[0].client_state is None


class TestVerifyCli:
    def test_parser_exposes_matrix_style_flags(self) -> None:
        args = _verify.build_parser().parse_args(
            ["--tracking-uri", "sqlite:///x.db", "--dataset", "mnist",
             "--method", "pldpbo_snr", "--seeds", "0-3"],
        )
        assert args.tracking_uri == "sqlite:///x.db"
        assert args.dataset == ["mnist"]
        assert args.method == ["pldpbo_snr"]
        assert args.seeds == "0-3"


def _vr(
    method: str = "pldpbo_snr",
    state: dict | None = None,
    params: dict[str, str] | None = None,
) -> object:
    return _verify.VerifyRun("mnist_iid", method, "runid", 0, params or {}, state)


class TestVerifyWarmup:
    def test_passes_with_grid_values(self) -> None:
        from src.privacy.bo_scheduler import WARMUP_GRID

        grid = list(WARMUP_GRID)
        state = {
            "0": {"acct_cost": grid, "r_t_final": grid},
            "1": {"acct_cost": grid, "r_t_final": grid},
        }
        result = _verify._check_warmup(_vr(state=state))
        assert result["pass"] is True
        assert result["n"] == 2
        assert abs(result["sum_mean"] - _verify.WARMUP_SUM_NOMINAL) < 1e-9
        assert result["parity_median"] == 0.0
        assert result["parity_max"] == 0.0

    def test_fails_out_of_tolerance(self) -> None:
        state = {"0": {"acct_cost": [0.2] * 10, "r_t_final": [0.2] * 10}}
        result = _verify._check_warmup(_vr(state=state))
        assert result["pass"] is False

    def test_uses_first_ten_participations_only(self) -> None:
        from src.privacy.bo_scheduler import WARMUP_GRID

        grid = list(WARMUP_GRID)
        state = {"0": {"acct_cost": grid + [5.0, 5.0], "r_t_final": grid + [5.0, 5.0]}}
        result = _verify._check_warmup(_vr(state=state))
        assert abs(result["sum_mean"] - _verify.WARMUP_SUM_NOMINAL) < 1e-9

    def test_sums_available_participations_when_fewer_than_ten(self) -> None:
        state = {"0": {"acct_cost": [0.1, 0.2, 0.3], "r_t_final": [0.1, 0.2, 0.3]}}
        result = _verify._check_warmup(_vr(state=state))
        assert result["n"] == 1
        assert abs(result["sum_mean"] - 0.6) < 1e-9

    def test_parity_excludes_refused_rounds(self) -> None:
        # r_t_final == 0.0 marks a refused round; excluded from parity.
        state = {
            "0": {
                "acct_cost": [1.0, 1.0, 0.0],
                "r_t_final": [1.5, 0.5, 0.0],
            },
        }
        result = _verify._check_warmup(_vr(state=state))
        # relative errors: |1-1.5|/1.5 = 1/3, |1-0.5|/0.5 = 1
        assert result["parity_median"] == pytest.approx(2 / 3)
        assert result["parity_max"] == pytest.approx(1.0)

    def test_no_data_gives_none(self) -> None:
        result = _verify._check_warmup(_vr(state=None))
        assert result["pass"] is None
        assert result["n"] == 0
