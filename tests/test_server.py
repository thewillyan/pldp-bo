from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.loader import ExperimentConfig
from src.server.strategy import _EPS, _MIN_VALUES_FOR_STATS


def _median_weights(norms: list[float]) -> np.ndarray:
    b = float(np.median(norms))
    weights = np.array(
        [1.0 if r <= 1e-12 else min(1.0, b / r) for r in norms],
        dtype=np.float64,
    )
    return weights


def _weighted_average(deltas: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    total = np.sum(weights)
    if total <= 0:
        return np.zeros_like(deltas[0])
    norm_w = weights / total
    return np.sum([w * d for w, d in zip(norm_w, deltas, strict=True)], axis=0)


class TestMedianWeightComputation:
    def test_median_with_outlier(self) -> None:
        norms = [1.0, 2.0, 10.0]
        weights = _median_weights(norms)
        expected = np.array([1.0, 1.0, 0.2])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_all_equal(self) -> None:
        norms = [3.0, 3.0, 3.0]
        weights = _median_weights(norms)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_single_client(self) -> None:
        norms = [5.0]
        weights = _median_weights(norms)
        expected = np.array([1.0])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_two_clients(self) -> None:
        norms = [1.0, 4.0]
        weights = _median_weights(norms)
        expected = np.array([1.0, 0.625])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_zero_norm_does_not_cause_division_error(self) -> None:
        norms = [0.0, 2.0, 8.0]
        weights = _median_weights(norms)
        assert np.all(np.isfinite(weights))

    def test_larger_norm_gets_smaller_weight(self) -> None:
        norms = [1.0, 5.0, 10.0]
        weights = _median_weights(norms)
        assert weights[1] > weights[2]


class TestWeightedAveraging:
    def test_simple_weighted_average(self) -> None:
        deltas = [np.array([1.0, 0.0]), np.array([0.0, 2.0])]
        weights = np.array([0.75, 0.25])
        result = _weighted_average(deltas, weights)
        expected = np.array([0.75, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_delta(self) -> None:
        deltas = [np.array([3.0, 4.0])]
        weights = np.array([1.0])
        result = _weighted_average(deltas, weights)
        np.testing.assert_array_almost_equal(result, [3.0, 4.0])

    def test_equal_weights(self) -> None:
        deltas = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        weights = np.array([1.0, 1.0])
        result = _weighted_average(deltas, weights)
        expected = np.array([2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_weights_returns_zeros(self) -> None:
        deltas = [np.array([1.0, 2.0])]
        weights = np.array([0.0])
        result = _weighted_average(deltas, weights)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0])

    def test_integration_with_median_weights(self) -> None:
        norms = [1.0, 2.0, 10.0]
        weights = _median_weights(norms)
        deltas = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
        agg = _weighted_average(deltas, weights)
        w_sum = (
            1.0 * np.array([1.0, 0.0])
            + 1.0 * np.array([0.0, 1.0])
            + 0.2 * np.array([0.5, 0.5])
        ) / 2.2
        expected = w_sum
        np.testing.assert_array_almost_equal(agg, expected)

    def test_server_learning_rate_scales_delta(self) -> None:
        current = np.array([1.0, 1.0])
        aggregated = np.array([1.5, 0.7])
        lr = 0.5
        delta = aggregated - current  # [0.5, -0.3]
        scaled = current + lr * delta  # [1.25, 0.85]
        expected = np.array([1.25, 0.85])
        np.testing.assert_array_almost_equal(scaled, expected)

    def test_server_learning_rate_identity_when_one(self) -> None:
        current = np.array([1.0, 2.0, 3.0])
        aggregated = np.array([2.0, 3.0, 4.0])
        lr = 1.0
        scaled = current + lr * (aggregated - current)
        np.testing.assert_array_almost_equal(scaled, aggregated)


class TestFilterValidReplies:
    def test_filters_replies_without_metrics(self) -> None:
        from flwr.app import Message, MetricRecord, RecordDict

        replies = [
            Message(
                content=RecordDict({"metrics": MetricRecord({"num-examples": 5})}),
                message_type="train",
                dst_node_id=0,
            ),
            Message(
                content=RecordDict({}),
                message_type="train",
                dst_node_id=1,
            ),
        ]
        from src.server.strategy import _filter_valid_replies

        valid = _filter_valid_replies(replies)
        assert len(valid) == 1
        assert valid[0].metadata.dst_node_id == 0

    def test_filters_replies_with_zero_examples(self) -> None:
        from flwr.app import Message, MetricRecord, RecordDict

        replies = [
            Message(
                content=RecordDict({"metrics": MetricRecord({"num-examples": 0})}),
                message_type="train",
                dst_node_id=0,
            ),
            Message(
                content=RecordDict({"metrics": MetricRecord({"num-examples": 10})}),
                message_type="train",
                dst_node_id=1,
            ),
        ]
        from src.server.strategy import _filter_valid_replies

        valid = _filter_valid_replies(replies)
        assert len(valid) == 1
        assert valid[0].metadata.dst_node_id == 1

    def test_returns_empty_when_all_invalid(self) -> None:
        from flwr.app import Message, RecordDict

        replies = [
            Message(content=RecordDict({}), message_type="train", dst_node_id=0),
            Message(content=RecordDict({}), message_type="train", dst_node_id=1),
        ]
        from src.server.strategy import _filter_valid_replies

        valid = _filter_valid_replies(replies)
        assert len(valid) == 0


class TestModuleConstants:
    def test_eps_is_small_positive(self) -> None:
        assert _EPS > 0
        assert _EPS < 1e-10

    def test_min_values_for_stats(self) -> None:
        assert _MIN_VALUES_FOR_STATS == 3


class _RecordingTracker:
    """Minimal tracker stub capturing logged metrics and artifact contents."""

    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, float], int]] = []
        self.artifacts: list[str] = []

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.metrics.append((dict(metrics), step if step is not None else 0))

    def log_artifact(self, local_path: str) -> None:
        with open(local_path) as f:
            self.artifacts.append(f.read())


class _FakeModel:
    def __init__(self, net: nn.Module) -> None:
        self._net = net

    def get_model(self) -> nn.Module:
        return self._net

    def set_weights(self, parameters: list[Any]) -> None:
        pass


def _linear_two_input_model() -> nn.Module:
    """Linear(2, 3) that classifies e0 -> 0 and e1 -> 1."""
    net = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        net.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    return net


class TestMacroF1:
    def test_hand_computed_three_class(self) -> None:
        from src.server_app import _macro_f1
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 1, 1]
        # class 0: tp=1, fp=0, fn=1 -> P=1.0, R=0.5, F=2/3
        # class 1: tp=2, fp=1, fn=0 -> P=2/3, R=1.0, F=0.8
        # class 2: tp=1, fp=0, fn=0 -> P=1.0, R=1.0, F=1.0
        assert _macro_f1(y_true, y_pred) == pytest.approx((2 / 3 + 0.8 + 1.0) / 3)

    def test_never_correct_class_contributes_zero(self) -> None:
        from src.server_app import _macro_f1
        # class 0: tp=0, fp=0, fn=1 -> F=0; class 1: tp=1, fp=1, fn=0 -> F=2/3
        assert _macro_f1([0, 1], [1, 1]) == pytest.approx((0.0 + 2 / 3) / 2)

    def test_all_correct_single_class(self) -> None:
        from src.server_app import _macro_f1
        assert _macro_f1([2, 2, 2], [2, 2, 2]) == pytest.approx(1.0)

    def test_empty_returns_zero(self) -> None:
        from src.server_app import _macro_f1
        assert _macro_f1([], []) == 0.0


class TestGlobalTestEvaluate:
    def _make_loader(self, x: torch.Tensor, y: torch.Tensor) -> DataLoader[Any]:
        return DataLoader(TensorDataset(x, y), batch_size=2)

    def test_round_zero_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.server_app import _run_global_test_evaluate
        calls: list[int] = []

        def _counting_loader(_cfg: object) -> DataLoader[Any]:
            calls.append(1)
            return self._make_loader(torch.eye(2), torch.tensor([0, 1]))

        monkeypatch.setattr("src.server_app.create_test_loader", _counting_loader)
        tracker = _RecordingTracker()
        result = _run_global_test_evaluate(
            0,
            ArrayRecord({}),
            ExperimentConfig(),
            cast(Any, _FakeModel(_linear_two_input_model())),
            cast(Any, tracker),
        )
        assert result is None
        assert calls == []
        assert tracker.metrics == []

    def test_logs_acc_test_and_f1_test_at_round_step(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.server_app import _run_global_test_evaluate
        monkeypatch.setattr(
            "src.server_app.create_test_loader",
            lambda _cfg: self._make_loader(torch.eye(2), torch.tensor([0, 1])),
        )
        tracker = _RecordingTracker()
        result = _run_global_test_evaluate(
            7,
            ArrayRecord({}),
            ExperimentConfig(),
            cast(Any, _FakeModel(_linear_two_input_model())),
            cast(Any, tracker),
        )
        assert result is not None
        assert result["acc_test"] == pytest.approx(1.0)
        assert result["f1_test"] == pytest.approx(1.0)
        assert tracker.metrics == [(  # type: ignore[comparison-overlap]
            {"acc_test": pytest.approx(1.0), "f1_test": pytest.approx(1.0)},
            7,
        )]

    def test_macro_f1_matches_hand_computed_on_mixed_batch(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.server_app import _run_global_test_evaluate
        x = torch.cat([torch.eye(2), torch.eye(2)[:1], torch.eye(2)[:1]])  # e0, e1, e0, e0
        y = torch.tensor([0, 1, 0, 1])
        monkeypatch.setattr(
            "src.server_app.create_test_loader",
            lambda _cfg: self._make_loader(x, y),
        )
        tracker = _RecordingTracker()
        result = _run_global_test_evaluate(
            3,
            ArrayRecord({}),
            ExperimentConfig(),
            cast(Any, _FakeModel(_linear_two_input_model())),
            cast(Any, tracker),
        )
        # predictions: [0, 1, 0, 0]
        assert result is not None
        assert result["acc_test"] == pytest.approx(0.75)
        assert result["f1_test"] == pytest.approx((0.8 + 2 / 3) / 2)


class _FakeGrid:
    def __init__(self, replies: list[Message]) -> None:
        self._replies = replies
        self.sent: list[Message] = []

    def get_node_ids(self) -> list[int]:
        return list(range(len(self._replies)))

    def send_and_receive(
        self, messages: list[Message], timeout: float | None = None,  # noqa: ARG002
    ) -> list[Message]:
        self.sent = messages
        return self._replies


def _test_accuracy_reply(msg: Message, pid: int, acc: float, n: int) -> Message:
    return Message(
        content=RecordDict({
            "config": ConfigRecord({
                "partition_id": pid,
                "test_accuracy": acc,
                "n_test": n,
            }),
        }),
        reply_to=msg,
    )


class TestFemnistClientTestAccuracy:
    def test_sends_query_with_task_and_final_arrays(self) -> None:
        from src.server_app import _run_femnist_client_test_accuracy
        msg = Message(content=RecordDict({}), message_type="query", dst_node_id=0)
        grid = _FakeGrid([_test_accuracy_reply(msg, 0, 0.9, 5)])
        tracker = _RecordingTracker()
        _run_femnist_client_test_accuracy(cast(Any, grid), ArrayRecord({}), cast(Any, tracker))
        assert len(grid.sent) == 1
        task = grid.sent[0].content.config_records.get("config", ConfigRecord()).get("task")
        assert task == "client_test_accuracy"
        assert isinstance(grid.sent[0].content.get("arrays"), ArrayRecord)

    def test_writes_deterministic_json_keyed_by_partition_id(self) -> None:
        from src.server_app import _run_femnist_client_test_accuracy
        msgs = [
            Message(content=RecordDict({}), message_type="query", dst_node_id=0),
            Message(content=RecordDict({}), message_type="query", dst_node_id=1),
        ]
        grid = _FakeGrid([
            _test_accuracy_reply(msgs[0], 2, 0.85, 10),
            _test_accuracy_reply(msgs[1], 1, 0.90, 5),
        ])
        tracker = _RecordingTracker()
        _run_femnist_client_test_accuracy(cast(Any, grid), ArrayRecord({}), cast(Any, tracker))
        assert len(tracker.artifacts) == 1
        payload = json.loads(tracker.artifacts[0])
        assert payload == {"client_test_acc": {"1": 0.90, "2": 0.85}}

    def test_missing_accuracy_reply_skipped(self) -> None:
        from src.server_app import _run_femnist_client_test_accuracy
        msg = Message(content=RecordDict({}), message_type="query", dst_node_id=0)
        grid = _FakeGrid([
            Message(
                content=RecordDict({
                    "config": ConfigRecord({"partition_id": 0}),
                }),
                reply_to=msg,
            ),
        ])
        tracker = _RecordingTracker()
        _run_femnist_client_test_accuracy(cast(Any, grid), ArrayRecord({}), cast(Any, tracker))
        assert tracker.artifacts == []
