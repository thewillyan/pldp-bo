from __future__ import annotations

from unittest.mock import patch

from src.config.loader import ExperimentConfig
from src.tracking.tracker import ExperimentTracker


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
        tracker = self._make_tracker()

        with patch("src.tracking.tracker.mlflow") as mock_mlflow:
            tracker.start_run()
            mock_mlflow.start_run.assert_called_once_with(run_name=None)
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
        tracker = self._make_tracker()

        with patch("src.tracking.tracker.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            result = ExperimentTracker.get_run_id()
            assert result is None
