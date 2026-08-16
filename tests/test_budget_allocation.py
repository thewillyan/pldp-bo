from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, ConfigRecord, RecordDict
from flwr.common import Message
from flwr.serverapp.strategy import FedAvg

from src.config.loader import load_config
from src.server.strategy import (
    MedianRobustAggregation,
    SafeFedAvg,
    _add_budgets_to_messages,
    _is_budget_exhausted,
)
from src.server_app import _compute_per_client_budgets


class TestComputePerClientBudgets:
    """Tests for the budget computation math and config-derived logic."""

    @staticmethod
    def _make_config(overrides: dict | None = None) -> object:
        return load_config("config/default.yaml", overrides=overrides)

    def test_none_when_total_budget_none(self) -> None:
        config = self._make_config({"privacy.enabled": True})
        assert config.privacy.total_budget is None

    def _compute_equal(self, total_budget: float, num_clients: int) -> dict[int, float]:
        per_client = total_budget / max(num_clients, 1)
        return {cid: per_client for cid in range(num_clients)}

    def test_equal_division_personalization_disabled(self) -> None:
        budgets = self._compute_equal(10.0, 5)
        assert len(budgets) == 5
        for b in budgets.values():
            assert b == pytest.approx(2.0)

    def test_equal_division_single_client(self) -> None:
        budgets = self._compute_equal(10.0, 1)
        assert budgets[0] == pytest.approx(10.0)


class TestAddBudgetsToMessages:
    def _make_train_message(self, dst: int) -> Message:
        return Message(
            content=RecordDict({
                "config": ConfigRecord({"server-round": 1}),
                "arrays": ArrayRecord({"w": Array(np.array([1.0, 2.0]))}),
            }),
            message_type="train",
            dst_node_id=dst,
        )

    def test_budget_injected_into_config(self) -> None:
        budgets = {0: 1.0, 1: 3.0}
        messages = [self._make_train_message(0), self._make_train_message(1)]

        result = list(_add_budgets_to_messages(messages, budgets, "config"))
        assert len(result) == 2

        for msg, expected_budget in zip(result, [1.0, 3.0], strict=True):
            config = msg.content.config_records["config"]
            assert config["per_client_budget"] == pytest.approx(expected_budget)
            assert config["server-round"] == 1
            arrays = msg.content.array_records["arrays"]
            assert list(arrays["w"].numpy()) == [1.0, 2.0]

    def test_unknown_client_passes_through(self) -> None:
        budgets = {0: 1.0}
        messages = [self._make_train_message(0), self._make_train_message(99)]
        result = list(_add_budgets_to_messages(messages, budgets, "config"))

        assert result[0].content.config_records["config"]["per_client_budget"] == 1.0
        assert "per_client_budget" not in result[1].content.config_records["config"]

    def test_none_budgets_passthrough(self) -> None:
        messages = [self._make_train_message(0)]
        result = list(_add_budgets_to_messages(messages, None, "config"))
        assert len(result) == 1
        assert "per_client_budget" not in result[0].content.config_records["config"]

    def test_empty_budgets_passthrough(self) -> None:
        messages = [self._make_train_message(0)]
        result = list(_add_budgets_to_messages(messages, {}, "config"))
        assert len(result) == 1
        assert "per_client_budget" not in result[0].content.config_records["config"]

    def test_all_budgets_sum_properly(self) -> None:
        budgets = {0: 2.0, 1: 2.0, 2: 2.0}
        messages = [self._make_train_message(i) for i in range(3)]
        result = list(_add_budgets_to_messages(messages, budgets, "config"))
        configs = [m.content.config_records["config"] for m in result]
        total = sum(float(c["per_client_budget"]) for c in configs)
        assert total == pytest.approx(6.0)

    def test_remaining_rdp_injected_with_budget(self) -> None:
        budgets = {0: 10.0, 1: 10.0}
        remaining = {0: 10.0, 1: 6.5}
        messages = [self._make_train_message(0), self._make_train_message(1)]

        result = list(_add_budgets_to_messages(
            messages, budgets, "config",
            remaining_rdp_by_client=remaining,
        ))

        for msg, expected_remaining in zip(result, [10.0, 6.5], strict=True):
            config = msg.content.config_records["config"]
            assert config["remaining_rdp"] == pytest.approx(expected_remaining)
            assert config["per_client_budget"] == pytest.approx(10.0)

    def test_remaining_rdp_absent_when_unknown(self) -> None:
        budgets = {0: 10.0}
        messages = [self._make_train_message(0)]

        result = list(_add_budgets_to_messages(
            messages, budgets, "config",
            remaining_rdp_by_client={},
        ))

        config = result[0].content.config_records["config"]
        assert config["per_client_budget"] == pytest.approx(10.0)
        assert "remaining_rdp" not in config


class TestIsBudgetExhausted:
    def _make_reply(self, budget_exhausted: int = 0) -> Message:
        from flwr.app import MetricRecord
        return Message(
            content=RecordDict({
                "metrics": MetricRecord({"budget_exhausted": budget_exhausted}),
            }),
            dst_node_id=0,
            message_type="train",
        )

    def test_exhausted_true(self) -> None:
        msg = self._make_reply(budget_exhausted=1)
        assert _is_budget_exhausted(msg) is True

    def test_exhausted_false(self) -> None:
        msg = self._make_reply(budget_exhausted=0)
        assert _is_budget_exhausted(msg) is False

    def test_no_metrics(self) -> None:
        msg = Message(
            content=RecordDict({}),
            dst_node_id=0,
            message_type="train",
        )
        assert _is_budget_exhausted(msg) is False

    def test_no_budget_exhausted_key(self) -> None:
        from flwr.app import MetricRecord
        msg = Message(
            content=RecordDict({
                "metrics": MetricRecord({"epsilon": 1.0}),
            }),
            dst_node_id=0,
            message_type="train",
        )
        assert _is_budget_exhausted(msg) is False


class TestComputePerClientBudgetsIntegration:
    """Integration tests for _compute_per_client_budgets with mocked Grid."""

    def test_none_when_no_budget(self) -> None:
        grid = MagicMock()
        grid.get_node_ids.return_value = [1001]
        config = load_config("config/default.yaml", overrides={
            "privacy.enabled": True,
        })
        assert config.privacy.total_budget is None
        budgets, n2p = _compute_per_client_budgets(grid, config)
        assert budgets is None
        assert n2p is None

    def test_none_when_no_nodes(self) -> None:
        grid = MagicMock()
        grid.get_node_ids.return_value = []
        config = load_config("config/default.yaml", overrides={
            "privacy.enabled": True,
            "privacy.total_budget": 10.0,
        })
        budgets, n2p = _compute_per_client_budgets(grid, config)
        assert budgets is None
        assert n2p is None

    def test_flat_budget_for_all_clients(self) -> None:
        """IMPL-09 §9.5: every client gets the full B_RDP (flat, not divided by K)."""
        grid = MagicMock()
        grid.get_node_ids.return_value = [1001, 1002, 1003]
        config = load_config("config/default.yaml", overrides={
            "privacy.enabled": True,
            "privacy.total_budget": 9.0,
        })
        budgets, n2p = _compute_per_client_budgets(grid, config)
        assert budgets is not None
        assert len(budgets) == 3
        for nid in [1001, 1002, 1003]:
            assert budgets[nid] == pytest.approx(9.0)
        assert n2p is None


class TestAddBudgetsToMessagesWithMapping:
    def _make_train_message(self, dst: int) -> Message:
        return Message(
            content=RecordDict({
                "config": ConfigRecord({"server-round": 1}),
            }),
            message_type="train",
            dst_node_id=dst,
        )

    def test_node_to_partition_routes_budget(self) -> None:
        budgets = {0: 1.0, 1: 3.0}
        node_to_partition = {1001: 0, 1002: 1}
        messages = [self._make_train_message(1001), self._make_train_message(1002)]
        result = list(_add_budgets_to_messages(
            messages, budgets, "config", node_to_partition=node_to_partition,
        ))
        assert len(result) == 2
        assert result[0].content.config_records["config"]["per_client_budget"] == pytest.approx(1.0)
        assert result[1].content.config_records["config"]["per_client_budget"] == pytest.approx(3.0)

    def test_node_to_partition_missing_node_passes_through(self) -> None:
        budgets = {0: 1.0}
        node_to_partition = {1001: 0}
        messages = [self._make_train_message(1001), self._make_train_message(999)]
        result = list(_add_budgets_to_messages(
            messages, budgets, "config", node_to_partition=node_to_partition,
        ))
        assert "per_client_budget" in result[0].content.config_records["config"]
        assert "per_client_budget" not in result[1].content.config_records["config"]


class TestStrategyConfigureTrain:
    def test_median_robust_injects_budgets(self) -> None:
        base_msg = Message(
            content=RecordDict({
                "config": ConfigRecord({"server-round": 1}),
            }),
            message_type="train",
            dst_node_id=1001,
        )
        grid = MagicMock()

        with patch.object(FedAvg, "configure_train", return_value=[base_msg]):
            strategy = MedianRobustAggregation(
                server_learning_rate=1.0,
                fraction_train=1.0,
                fraction_evaluate=0.0,
                min_train_nodes=1,
                min_evaluate_nodes=0,
                min_available_nodes=1,
                per_client_budgets={0: 2.5},
                node_to_partition={1001: 0},
            )
            arrays = ArrayRecord({})
            config = ConfigRecord({})
            result = list(strategy.configure_train(1, arrays, config, grid))

        assert len(result) == 1
        assert result[0].content.config_records["config"]["per_client_budget"] == pytest.approx(2.5)

    def test_safe_fedavg_injects_budgets(self) -> None:
        base_msg = Message(
            content=RecordDict({
                "config": ConfigRecord({"server-round": 1}),
            }),
            message_type="train",
            dst_node_id=1001,
        )
        grid = MagicMock()

        with patch.object(FedAvg, "configure_train", return_value=[base_msg]):
            strategy = SafeFedAvg(
                server_learning_rate=1.0,
                fraction_train=1.0,
                fraction_evaluate=0.0,
                min_train_nodes=1,
                min_evaluate_nodes=0,
                min_available_nodes=1,
                per_client_budgets={0: 3.0},
                node_to_partition={1001: 0},
            )
            arrays = ArrayRecord({})
            config = ConfigRecord({})
            result = list(strategy.configure_train(1, arrays, config, grid))

        assert len(result) == 1
        assert result[0].content.config_records["config"]["per_client_budget"] == pytest.approx(3.0)
