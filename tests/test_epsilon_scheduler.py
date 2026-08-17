from __future__ import annotations

import pytest

from src.privacy.epsilon_scheduler import (
    FixedEpsilonScheduler,
    UniformRandomEpsilonScheduler,
)


class TestFixedEpsilonScheduler:
    def test_always_returns_same_value(self) -> None:
        scheduler = FixedEpsilonScheduler(epsilon=2.5)
        for _ in range(10):
            assert scheduler.get_epsilon() == 2.5

    def test_step_is_noop(self) -> None:
        scheduler = FixedEpsilonScheduler(epsilon=1.0)
        scheduler.step(epsilon=1.0, metric=0.5)
        assert scheduler.get_epsilon() == 1.0

    def test_serialization_roundtrip(self) -> None:
        original = FixedEpsilonScheduler(epsilon=3.0)
        state = original.get_state()
        restored = FixedEpsilonScheduler.from_state(state)
        assert restored.get_epsilon() == 3.0
        assert restored.get_epsilon() == original.get_epsilon()

    def test_repr(self) -> None:
        scheduler = FixedEpsilonScheduler(epsilon=1.5)
        assert "FixedEpsilonScheduler" in repr(scheduler)
        assert "1.5" in repr(scheduler)

    def test_invalid_epsilon_raises(self) -> None:
        with pytest.raises(ValueError, match="epsilon must be positive"):
            FixedEpsilonScheduler(epsilon=0.0)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            FixedEpsilonScheduler(epsilon=-1.0)


class TestUniformRandomEpsilonScheduler:
    def test_values_in_bounds(self) -> None:
        scheduler = UniformRandomEpsilonScheduler(epsilon_min=1.0, epsilon_max=5.0, seed=42)
        for _ in range(100):
            eps = scheduler.get_epsilon()
            assert 1.0 <= eps <= 5.0

    def test_reproducible_with_seed(self) -> None:
        s1 = UniformRandomEpsilonScheduler(epsilon_min=0.5, epsilon_max=3.0, seed=123)
        s2 = UniformRandomEpsilonScheduler(epsilon_min=0.5, epsilon_max=3.0, seed=123)
        values1 = [s1.get_epsilon() for _ in range(10)]
        values2 = [s2.get_epsilon() for _ in range(10)]
        assert values1 == values2

    def test_different_seeds_differ(self) -> None:
        s1 = UniformRandomEpsilonScheduler(epsilon_min=0.5, epsilon_max=3.0, seed=1)
        s2 = UniformRandomEpsilonScheduler(epsilon_min=0.5, epsilon_max=3.0, seed=2)
        values1 = [s1.get_epsilon() for _ in range(10)]
        values2 = [s2.get_epsilon() for _ in range(10)]
        assert values1 != values2

    def test_serialization_roundtrip(self) -> None:
        original = UniformRandomEpsilonScheduler(epsilon_min=0.1, epsilon_max=10.0, seed=42)
        original.get_epsilon()
        original.get_epsilon()
        state = original.get_state()
        restored = UniformRandomEpsilonScheduler.from_state(state)
        assert restored.get_epsilon() == original.get_epsilon()

    def test_step_is_noop(self) -> None:
        scheduler = UniformRandomEpsilonScheduler(epsilon_min=1.0, epsilon_max=2.0, seed=42)
        val_before = scheduler.get_epsilon()
        scheduler.step(epsilon=val_before, metric=0.5)
        assert scheduler.get_epsilon() != val_before

    def test_repr(self) -> None:
        scheduler = UniformRandomEpsilonScheduler(epsilon_min=0.5, epsilon_max=8.0)
        assert "UniformRandomEpsilonScheduler" in repr(scheduler)
        assert "0.5" in repr(scheduler)
        assert "8.0" in repr(scheduler)

    def test_invalid_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="epsilon_min must be positive"):
            UniformRandomEpsilonScheduler(epsilon_min=0.0, epsilon_max=1.0)
        with pytest.raises(ValueError, match="epsilon_max must be greater"):
            UniformRandomEpsilonScheduler(epsilon_min=5.0, epsilon_max=3.0)
        with pytest.raises(ValueError, match="epsilon_max must be greater"):
            UniformRandomEpsilonScheduler(epsilon_min=2.0, epsilon_max=2.0)
