from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_script(path: str, module_name: str) -> types.ModuleType:
    import importlib.machinery
    loader = importlib.machinery.SourceFileLoader(module_name, path)
    mod = types.ModuleType(module_name)
    mod.__file__ = path
    loader.exec_module(mod)
    return mod


_run = _load_script("scripts/run", "_run_script")
_plot = _load_script("scripts/plot", "_plot_script")


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
        _, depth = _run._parse_json_from_stream('{\n', buffer, 0)
        assert depth == 1
        data, depth = _run._parse_json_from_stream('  "a": 1\n', buffer, 1)
        assert depth == 1
        data, depth = _run._parse_json_from_stream('}\n', buffer, 1)
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
            _run.mlflow.tracking, "MlflowClient", lambda: mock_client,
        )
        run_id = _run._resolve_run_id("sqlite:///test.db", "my-run")
        assert run_id == "abc123"
        _, kwargs = mock_client.search_runs.call_args
        assert "my-run" in kwargs["filter_string"]

    def test_returns_none_when_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client.search_runs.return_value = []
        monkeypatch.setattr(
            _run.mlflow.tracking, "MlflowClient", lambda: mock_client,
        )
        run_id = _run._resolve_run_id("sqlite:///test.db", "missing-run")
        assert run_id is None

    def test_escapes_run_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client.search_runs.return_value = []
        monkeypatch.setattr(
            _run.mlflow.tracking, "MlflowClient", lambda: mock_client,
        )
        _run._resolve_run_id("sqlite:///test.db", "test's run")
        _, kwargs = mock_client.search_runs.call_args
        assert "test's run" not in kwargs["filter_string"]
        assert "test" in kwargs["filter_string"]

    def test_escapes_backslash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client.search_runs.return_value = []
        monkeypatch.setattr(
            _run.mlflow.tracking, "MlflowClient", lambda: mock_client,
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
