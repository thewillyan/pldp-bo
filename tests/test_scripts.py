from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.config.loader import load_config
from src.config.locked import collect_violations
from src.config.locked import config_version as locked_config_version


def _load_script(path: str, module_name: str) -> types.ModuleType:
    import importlib.machinery

    loader = importlib.machinery.SourceFileLoader(module_name, path)
    mod = types.ModuleType(module_name)
    mod.__file__ = path
    loader.exec_module(mod)
    return mod


_run = _load_script("scripts/run", "_run_script")
_plot = _load_script("scripts/plot", "_plot_script")
_gen = _load_script("scripts/gen_matrix_configs", "_gen_script")


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
