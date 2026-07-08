from __future__ import annotations

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, ConfigRecord, RecordDict
from flwr.common import Message

from src.config.loader import load_config
from src.server.strategy import _add_budgets_to_messages, _is_budget_exhausted


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

    def test_custom_proportional_division(self) -> None:
        eps_map = {"0": 1.0, "1": 2.0, "2": 3.0, "3": 4.0}
        total_budget = 20.0
        total_weight = sum(eps_map.values())
        budgets = {int(k): total_budget * v / total_weight for k, v in eps_map.items()}

        assert budgets[0] == pytest.approx(2.0)
        assert budgets[1] == pytest.approx(4.0)
        assert budgets[2] == pytest.approx(6.0)
        assert budgets[3] == pytest.approx(8.0)
        assert sum(budgets.values()) == pytest.approx(20.0)

    def test_custom_proportional_from_config(self) -> None:
        config = self._make_config({
            "privacy.enabled": True,
            "privacy.total_budget": 10.0,
            "personalization.enabled": True,
            "personalization.strategy": "custom",
            "personalization.client_epsilon_map": {"0": 1.0, "1": 3.0},
        })
        assert config.privacy.total_budget == 10.0
        eps_map = config.personalization.client_epsilon_map
        total_weight = sum(eps_map.values())
        budgets = {int(k): 10.0 * v / total_weight for k, v in eps_map.items()}
        assert budgets[0] == pytest.approx(2.5)
        assert budgets[1] == pytest.approx(7.5)
        assert sum(budgets.values()) == pytest.approx(10.0)

    def test_equal_division_custom_zero_weight(self) -> None:
        cfg = {
            "privacy.enabled": True,
            "privacy.total_budget": 10.0,
            "personalization.enabled": True,
            "personalization.strategy": "custom",
            "personalization.client_epsilon_map": {"0": 0.0, "1": 0.0},
            "data.num_clients": 2,
        }
        config = self._make_config(cfg)
        per_client = config.privacy.total_budget / config.data.num_clients
        assert per_client == 5.0


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
