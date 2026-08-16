from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pytest
import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.loader import ExperimentConfig
from src.server.strategy import _EPS, _MIN_VALUES_FOR_STATS
from src.tracking.tracker import ExperimentTracker


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


class TestRemainingRdpMap:
    def _make_strategy(self) -> Any:
        from src.server.strategy import MedianRobustAggregation
        return MedianRobustAggregation(
            per_client_budgets={0: 10.0, 1: 10.0},
            node_to_partition={7: 0, 8: 1},
        )

    def test_round_one_full_budget(self) -> None:
        strategy = self._make_strategy()
        assert strategy._remaining_rdp_map() == {0: 10.0, 1: 10.0}

    def test_subtracts_last_reported_cum_rdp(self) -> None:
        strategy = self._make_strategy()
        strategy._client_cum_rdp = {0: 4.0, 1: 10.0}
        assert strategy._remaining_rdp_map() == {0: 6.0, 1: 0.0}

    def test_falls_back_to_cumulative_epsilon(self) -> None:
        strategy = self._make_strategy()
        strategy._client_cum_eps = {0: 3.0}
        assert strategy._remaining_rdp_map() == {0: 7.0, 1: 10.0}

    def test_node_keyed_budgets_resolve_partitions(self) -> None:
        from src.server.strategy import MedianRobustAggregation
        strategy = MedianRobustAggregation(
            per_client_budgets={7: 10.0, 8: 10.0},
            node_to_partition={7: 0, 8: 1},
        )
        strategy._client_cum_rdp = {0: 2.5}
        assert strategy._remaining_rdp_map() == {7: 7.5, 8: 10.0}

    def test_none_when_no_budgets(self) -> None:
        from src.server.strategy import MedianRobustAggregation
        strategy = MedianRobustAggregation()
        assert strategy._remaining_rdp_map() is None


def _metrics_content(fields: dict[str, object]) -> RecordDict:
    return RecordDict({"metrics": MetricRecord(fields)})


def _logged_map(tracker: Any) -> dict[str, tuple[Any, int]]:
    result: dict[str, tuple[Any, int]] = {}
    for metrics, step in tracker.metrics:
        for key, value in metrics.items():
            result[key] = (value, step)
    return result


class _FakeGridNodes:
    def __init__(self, node_ids: list[int]) -> None:
        self._node_ids = node_ids

    def get_node_ids(self) -> list[int]:
        return self._node_ids


class TestSpecRoundMetrics:
    """IMPL-11 §4.3 per-round metrics logged by the strategy."""

    def _make_strategy(self) -> Any:
        from src.server.strategy import MedianRobustAggregation

        return MedianRobustAggregation(
            tracker=_RecordingTracker(),
            per_client_budgets={0: 10.0, 1: 10.0},
        )

    def _active_contents(self) -> list[RecordDict]:
        return [
            _metrics_content({
                "client-id": 0,
                "num-examples": 4,
                "r_t_final": 0.5,
                "cumulative_rdp": 1.0,
                "bo_time": 0.1,
                "acct_time": 0.02,
            }),
            _metrics_content({
                "client-id": 1,
                "num-examples": 4,
                "r_t_final": 0.3,
                "cumulative_rdp": 0.6,
                "bo_time": 0.2,
                "acct_time": 0.04,
            }),
        ]

    def test_logs_section43_keys_at_round_step(self) -> None:
        strategy = self._make_strategy()
        strategy._bytes_sent_round = 100
        strategy._log_client_metrics(7, self._active_contents())
        logged = _logged_map(strategy._tracker)
        assert logged["n_participants"] == (2.0, 7)
        assert logged["mean_r_t"] == (0.4, 7)
        assert logged["mean_cum_rdp"] == (0.8, 7)
        assert logged["bo_time_round"] == (pytest.approx(0.15), 7)
        assert logged["acct_time_round"] == (pytest.approx(0.03), 7)
        assert logged["bytes_round"] == (100.0, 7)

    def test_bytes_round_counts_received_arrays(self) -> None:
        strategy = self._make_strategy()
        strategy._bytes_sent_round = 32
        contents = [_metrics_content({"client-id": 0, "num-examples": 4})]
        contents[0]["arrays"] = ArrayRecord({"w": Array(np.zeros((2, 2), dtype=np.float64))})
        strategy._log_client_metrics(1, contents)
        logged = _logged_map(strategy._tracker)
        assert logged["bytes_round"] == (32.0 + 32.0, 1)

    def test_configure_train_counts_sent_bytes(self) -> None:
        strategy = self._make_strategy()
        arrays = ArrayRecord({"w": Array(np.zeros((2, 2), dtype=np.float64))})
        messages = list(
            strategy.configure_train(1, arrays, ConfigRecord({}), _FakeGridNodes([0, 1]))
        )
        assert len(messages) == 2
        assert strategy._bytes_sent_round == 2 * 32


class TestClientStateAccumulation:
    """IMPL-11 §4.4 per-client per-participation accumulation."""

    def _make_strategy(self) -> Any:
        from src.server.strategy import MedianRobustAggregation

        return MedianRobustAggregation(
            tracker=_RecordingTracker(),
            per_client_budgets={0: 10.0, 1: 10.0},
        )

    def test_accumulates_participations(self) -> None:
        strategy = self._make_strategy()
        strategy._remaining_rdp_sent = {0: 9.4, 1: 9.0}
        strategy._log_client_metrics(
            1,
            [
                _metrics_content({
                    "client-id": 0,
                    "num-examples": 4,
                    "r_t_candidate": 0.6,
                    "r_t_final": 0.5,
                    "cumulative_rdp": 1.0,
                    "phase": 0.0,
                    "observed_m": 0.42,
                    "acct_cost": 0.31,
                    "utility_loss_clean": 1.2,
                    "utility_loss": 1.4,
                    "update_norm": 0.7,
                    "update_norm_clean": 0.5,
                    "sigma": 8.0,
                    "logit_disagreement": 0.1,
                }),
            ],
        )
        s = strategy._client_state[0]
        assert s["r_t_candidate"] == [0.6]
        assert s["r_t_final"] == [0.5]
        assert s["cum_rdp"] == [1.0]
        assert s["remaining_rdp"] == [9.4]
        assert s["phase"] == ["warmup"]
        assert s["observed_m"] == [0.42]
        assert s["acct_cost"] == [0.31]
        assert s["L_clean"] == [1.2]
        assert s["L_noisy"] == [1.4]
        assert s["update_norm_noisy"] == [0.7]
        assert s["update_norm_clean"] == [0.5]
        assert s["sigma"] == [8.0]
        assert s["agreement"] == [0.1]
        assert s["warmup_rounds"] == [1]
        assert s["enforcement_count"] == 1

    def test_warmup_rounds_and_enforcement_across_rounds(self) -> None:
        strategy = self._make_strategy()
        strategy._remaining_rdp_sent = {0: 9.0}
        strategy._log_client_metrics(
            1,
            [
                _metrics_content({
                    "client-id": 0,
                    "num-examples": 4,
                    "r_t_candidate": 0.05,
                    "r_t_final": 0.02,
                    "cumulative_rdp": 0.5,
                    "phase": 0.0,
                    "acct_cost": 0.5,
                }),
            ],
        )
        strategy._log_client_metrics(
            2,
            [
                _metrics_content({
                    "client-id": 0,
                    "num-examples": 4,
                    "r_t_candidate": 0.4,
                    "r_t_final": 0.4,
                    "cumulative_rdp": 0.9,
                    "phase": 1.0,
                    "acct_cost": 0.1,
                }),
            ],
        )
        s = strategy._client_state[0]
        assert s["phase"] == ["warmup", "bo"]
        assert s["warmup_rounds"] == [1]
        assert s["enforcement_count"] == 1

    def test_exhausted_participations_and_dropout_round(self) -> None:
        strategy = self._make_strategy()
        strategy._remaining_rdp_sent = {0: 0.0}
        reply = Message(
            content=_metrics_content({
                "client-id": 0,
                "num-examples": 0,
                "budget_exhausted": 1.0,
                "rdp_cost": 0.0,
                "cumulative_rdp": 10.0,
                "phase": 2.0,
                "r_t_candidate": 0.05,
                "update_norm": 0.0,
                "sigma": 0.0,
            }),
            message_type="train",
            dst_node_id=0,
        )
        strategy._record_exhausted(5, [reply])
        assert strategy._client_dropout_round == {0: 5}
        s = strategy._client_state[0]
        assert s["phase"] == ["exhausted"]
        assert s["r_t_final"] == [0.0]
        assert s["acct_cost"] == [0.0]
        assert s["cum_rdp"] == [10.0]
        assert s["remaining_rdp"] == [0.0]
        assert s["r_t_candidate"] == [0.05]
        assert s["observed_m"] == [None]
        assert s["warmup_rounds"] == []
        assert s["enforcement_count"] == 0

    def test_get_client_state_includes_dropout_default(self) -> None:
        strategy = self._make_strategy()
        strategy._remaining_rdp_sent = {0: 9.0}
        strategy._log_client_metrics(
            1,
            [
                _metrics_content({
                    "client-id": 0,
                    "num-examples": 4,
                    "r_t_candidate": 0.5,
                    "r_t_final": 0.5,
                    "cumulative_rdp": 1.0,
                    "phase": 1.0,
                    "acct_cost": 0.2,
                }),
            ],
        )
        state = strategy.get_client_state()
        assert state[0]["dropout_round"] is None
        assert state[0]["enforcement_count"] == 0

    def test_nonprivate_contents_do_not_accumulate(self) -> None:
        strategy = self._make_strategy()
        strategy._log_client_metrics(
            1,
            [_metrics_content({"client-id": 0, "num-examples": 4, "update_norm": 0.5})],
        )
        assert strategy._client_state == {}

    def test_bo_acct_time_totals(self) -> None:
        strategy = self._make_strategy()
        strategy._log_client_metrics(
            1,
            [_metrics_content({
                "client-id": 0, "num-examples": 4, "bo_time": 0.1, "acct_time": 0.02,
            })],
        )
        strategy._log_client_metrics(
            2,
            [_metrics_content({
                "client-id": 0, "num-examples": 4, "bo_time": 0.3, "acct_time": 0.04,
            })],
        )
        assert strategy.get_bo_time_total() == pytest.approx(0.4)
        assert strategy.get_acct_time_total() == pytest.approx(0.06)


class TestClientStateArtifact:
    """IMPL-11 §4.4 final artifact + §4.3 final metrics."""

    def _config(self, *, privacy_enabled: bool = True) -> ExperimentConfig:
        cfg = ExperimentConfig()
        cfg.privacy.enabled = privacy_enabled
        cfg.privacy.total_budget = 10.0
        return cfg

    def _accumulated_strategy(self) -> Any:
        from src.server.strategy import MedianRobustAggregation

        tracker = _RecordingTracker()
        strategy = MedianRobustAggregation(tracker=tracker, per_client_budgets={0: 10.0})
        strategy._remaining_rdp_sent = {0: 9.0}
        strategy._log_client_metrics(
            1,
            [
                _metrics_content({
                    "client-id": 0,
                    "num-examples": 4,
                    "r_t_candidate": 0.05,
                    "r_t_final": 0.05,
                    "cumulative_rdp": 2.0,
                    "phase": 1.0,
                    "acct_cost": 0.2,
                }),
            ],
        )
        return strategy

    def test_writes_artifact_and_final_metrics(self) -> None:
        from src.server_app import _write_client_state_artifact

        tracker = _RecordingTracker()
        strategy = self._accumulated_strategy()
        strategy._tracker = tracker
        _write_client_state_artifact(strategy, self._config(), tracker, 10.0, 3)
        payload = json.loads(tracker.artifacts[0])
        assert "client_state" in payload
        entry = payload["client_state"]["0"]
        assert entry["r_t_candidate"] == [0.05]
        assert entry["r_t_final"] == [0.05]
        assert entry["cum_rdp"] == [2.0]
        assert entry["remaining_rdp"] == [9.0]
        assert entry["phase"] == ["bo"]
        assert entry["dropout_round"] is None
        assert entry["enforcement_count"] == 0
        metrics = _logged_map(tracker)
        assert metrics["budget_utilization"] == (pytest.approx(0.2), 3)
        assert metrics["bo_overhead_pct"] == (pytest.approx(0.0), 3)

    def test_bo_overhead_pct_computed(self) -> None:
        from src.server_app import _write_client_state_artifact

        tracker = _RecordingTracker()
        strategy = self._accumulated_strategy()
        strategy._tracker = tracker
        strategy._bo_time_total = 0.5
        _write_client_state_artifact(strategy, self._config(), tracker, 10.0, 3)
        metrics = _logged_map(tracker)
        assert metrics["bo_overhead_pct"] == (pytest.approx(5.0), 3)

    def test_nonprivate_skips_artifact(self) -> None:
        from src.server_app import _write_client_state_artifact

        tracker = _RecordingTracker()
        strategy = self._accumulated_strategy()
        strategy._tracker = tracker
        _write_client_state_artifact(
            strategy, self._config(privacy_enabled=False), tracker, 10.0, 3,
        )
        assert tracker.artifacts == []

    def test_skips_artifact_without_state(self) -> None:
        from src.server.strategy import MedianRobustAggregation
        from src.server_app import _write_client_state_artifact

        tracker = _RecordingTracker()
        strategy = MedianRobustAggregation(tracker=tracker, per_client_budgets={0: 10.0})
        _write_client_state_artifact(strategy, self._config(), tracker, 10.0, 3)
        assert tracker.artifacts == []
        assert tracker.metrics == []


class TestStrategyRouting:
    """IMPL-09 §9.5: server strategy selected from federated.aggregation."""

    @staticmethod
    def _config(aggregation: str, method: str) -> ExperimentConfig:
        cfg = ExperimentConfig()
        cfg.federated.aggregation = aggregation
        cfg.method = method
        return cfg

    def test_plain_gets_safe_fed_avg(self) -> None:
        from src.server.strategy import SafeFedAvg
        from src.server_app import _make_strategy

        strategy = _make_strategy(self._config("plain", "nonprivate"), None, None, None)
        assert isinstance(strategy, SafeFedAvg)

    def test_attenuation_gets_median_for_all_private_methods(self) -> None:
        from src.server.strategy import MedianRobustAggregation
        from src.server_app import _make_strategy

        for method in ("dpfedavg_fixed", "fedprox_fixed", "pldpbo_nun"):
            strategy = _make_strategy(self._config("attenuation", method), None, None, None)
            assert isinstance(strategy, MedianRobustAggregation), method

    def test_median_strategy_uses_tracker_and_budgets(self) -> None:
        from src.server.strategy import MedianRobustAggregation
        from src.server_app import _make_strategy

        strategy = _make_strategy(
            self._config("attenuation", "pldpbo_nun"),
            cast("ExperimentTracker | None", _RecordingTracker()), {0: 10.0}, {0: 0},
        )
        assert isinstance(strategy, MedianRobustAggregation)
        assert strategy._tracker is not None
        assert strategy._per_client_budgets == {0: 10.0}
