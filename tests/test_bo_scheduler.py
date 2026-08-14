from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest

from src.config.loader import BOConfig
from src.privacy.bo_scheduler import (
    WARMUP_GRID,
    WARMUP_SUM_NOMINAL,
    PLDPBOScheduler,
    expected_improvement,
    normalize_ei,
)


class TestExpectedImprovement:
    def test_basic(self) -> None:
        mean = np.array([0.0, -1.0, 1.0])
        std = np.array([1.0, 1.0, 1.0])
        f_best = 0.0
        ei = expected_improvement(mean, std, f_best)
        # At mean=0, f_best=0 => EI = 0*Φ(0) + 1*φ(0) ≈ 0.3989
        assert ei[0] == pytest.approx(0.3989, abs=1e-3)
        # At mean=-1, improvement=1 => EI > ei[0]
        assert ei[1] > ei[0]
        # At mean=1 (worse than f_best), std=1 creates exploration term:
        # EI = (-1)*Φ(-1) + 1*φ(-1) ≈ 0.0833
        assert ei[2] == pytest.approx(0.0833, abs=1e-3)

    def test_degenerate_std(self) -> None:
        mean = np.array([0.0, 0.0, 0.0])
        std = np.array([0.0, 0.0, 0.0])
        f_best = 0.0
        ei = expected_improvement(mean, std, f_best)
        # With clamped std=1e-12 and f_best == mean:
        # EI = 0*Φ(0) + 1e-12*φ(0) ≈ 3.99e-13
        assert np.allclose(ei, 0.0, atol=1e-10)

    def test_best_seen_point(self) -> None:
        mean = np.array([0.5])
        std = np.array([0.1])
        f_best = 0.5
        ei = expected_improvement(mean, std, f_best)
        # f_best == mean, but std=0.1 gives exploration term:
        # EI = 0*Φ(0) + 0.1*φ(0) ≈ 0.0399
        assert ei[0] == pytest.approx(0.0399, abs=1e-3)


class TestNormalizeEi:
    def test_basic(self) -> None:
        ei = np.array([0.0, 1.0, 2.0])
        normalized = normalize_ei(ei)
        assert normalized[0] == 0.0
        assert normalized[2] == 1.0
        assert normalized[1] == 0.5

    def test_degenerate(self) -> None:
        ei = np.array([1.0, 1.0, 1.0])
        normalized = normalize_ei(ei)
        assert np.all(normalized == 0.0)

    def test_single_value(self) -> None:
        ei = np.array([5.0])
        normalized = normalize_ei(ei)
        assert normalized[0] == 0.0


class TestPLDPBOScheduler:
    EPS_MIN = 0.1
    EPS_MAX = 5.0
    WARMUP = 10

    def make_scheduler(
        self,
        warmup_rounds: int | None = None,
        seed: int = 42,
        **kwargs,
    ) -> PLDPBOScheduler:
        return PLDPBOScheduler(
            epsilon_min=self.EPS_MIN,
            epsilon_max=self.EPS_MAX,
            warmup_rounds=self.WARMUP if warmup_rounds is None else warmup_rounds,
            seed=seed,
            **kwargs,
        )

    # --- Warm-up phase ---

    def test_warmup_returns_sequentially(self) -> None:
        scheduler = self.make_scheduler()
        values = []
        for _ in range(self.WARMUP):
            values.append(scheduler.get_epsilon())
            scheduler.step(values[-1], 0.0)
        # Fixed log-spaced warm-up grid (spec §9.3), independent of the bounds
        np.testing.assert_array_almost_equal(values, WARMUP_GRID)

    def test_warmup_values_in_range(self) -> None:
        scheduler = self.make_scheduler()
        for _ in range(self.WARMUP):
            eps = scheduler.get_epsilon()
            scheduler.step(eps, 1.0)
            assert WARMUP_GRID[0] <= eps <= WARMUP_GRID[-1]

    def test_warmup_covers_full_range(self) -> None:
        scheduler = self.make_scheduler()
        values = []
        for _ in range(self.WARMUP):
            values.append(scheduler.get_epsilon())
            scheduler.step(values[-1], 0.0)
        assert values[0] == pytest.approx(WARMUP_GRID[0])
        assert values[-1] == pytest.approx(WARMUP_GRID[-1])

    def test_phase_transition_after_warmup(self) -> None:
        scheduler = self.make_scheduler()
        for i in range(self.WARMUP):
            eps = scheduler.get_epsilon()
            scheduler.step(eps, float(i) / self.WARMUP)
        assert scheduler._phase == "bo"

    def test_gp_fitted_after_transition(self) -> None:
        scheduler = self.make_scheduler()
        for i in range(self.WARMUP):
            eps = scheduler.get_epsilon()
            scheduler.step(eps, float(i) / self.WARMUP)
        assert scheduler._gp is not None
        grid = np.array([[0.5], [2.5], [4.5]])
        mean, std = scheduler._gp.predict(grid, return_std=True)
        assert len(mean) == 3
        assert len(std) == 3
        assert np.all(std >= 0)

    def test_bo_epsilon_in_range(self) -> None:
        scheduler = self.make_scheduler()
        for i in range(self.WARMUP):
            eps = scheduler.get_epsilon()
            scheduler.step(eps, float(i) / self.WARMUP)
        for _ in range(10):
            eps = scheduler.get_epsilon()
            assert self.EPS_MIN <= eps <= self.EPS_MAX

    def test_bo_prefers_lower_epsilon_with_constant_metric(self) -> None:
        # The search space must be covered by the fixed warm-up grid, or the
        # GP extrapolates real EI variation into the uncovered region and the
        # acquisition penalty cannot dominate.
        scheduler = PLDPBOScheduler(
            epsilon_min=WARMUP_GRID[0],
            epsilon_max=WARMUP_GRID[-1],
            warmup_rounds=self.WARMUP,
            acquisition_penalty=0.5,
            seed=42,
        )
        for _ in range(self.WARMUP):
            eps = scheduler.get_epsilon()
            scheduler.step(eps, 1.0)
        eps = scheduler.get_epsilon()
        assert eps == pytest.approx(WARMUP_GRID[0])

    # --- Serialization ---

    def test_serialization_warmup_roundtrip(self) -> None:
        scheduler = self.make_scheduler()
        eps0 = scheduler.get_epsilon()
        scheduler.step(eps0, 0.5)
        eps1 = scheduler.get_epsilon()
        state = scheduler.get_state()

        restored = PLDPBOScheduler.from_state(state)
        assert restored._phase == scheduler._phase
        assert restored._round == scheduler._round
        assert restored._observations == scheduler._observations
        assert restored.get_epsilon() == pytest.approx(eps1)

    def test_serialization_bo_roundtrip(self) -> None:
        scheduler = self.make_scheduler()
        for i in range(self.WARMUP):
            eps = scheduler.get_epsilon()
            scheduler.step(eps, float(i) / self.WARMUP)
        state = scheduler.get_state()

        restored = PLDPBOScheduler.from_state(state)
        assert restored._phase == "bo"
        assert restored._gp is not None
        eps_restored = restored.get_epsilon()
        assert self.EPS_MIN <= eps_restored <= self.EPS_MAX

    def test_serialization_full_roundtrip_values(self) -> None:
        scheduler = self.make_scheduler()
        for _ in range(self.WARMUP):
            eps = scheduler.get_epsilon()
            scheduler.step(eps, 1.0)
        state = scheduler.get_state()
        restored = PLDPBOScheduler.from_state(state)
        assert restored.get_epsilon() == pytest.approx(scheduler.get_epsilon())

    def test_serialization_preserves_kernel_params(self) -> None:
        scheduler = self.make_scheduler(warmup_rounds=5)
        for i in range(5):
            eps = scheduler.get_epsilon()
            scheduler.step(eps, float(i) / 5.0)
        state = scheduler.get_state()
        assert "gp_kernel_params" in state
        assert state["gp_kernel_params"] is not None

        restored = PLDPBOScheduler.from_state(state)
        # _restored_kernel is used (not consumed) by _fit_gp during from_state;
        # verify preservation by checking that GP predictions match
        np.testing.assert_array_almost_equal(
            restored._gp.predict(np.array([[0.5], [2.5]]), return_std=True)[0],
            scheduler._gp.predict(np.array([[0.5], [2.5]]), return_std=True)[0],
        )

    # --- RNG preservation ---

    def test_rng_preserved_after_serialization(self) -> None:
        s1 = self.make_scheduler(seed=42)
        for _ in range(self.WARMUP + 3):
            eps = s1.get_epsilon()
            s1.step(eps, float(eps))

        restored = PLDPBOScheduler.from_state(copy.deepcopy(s1.get_state()))
        for _ in range(5):
            e1 = s1.get_epsilon()
            e2 = restored.get_epsilon()
            assert e1 == pytest.approx(e2)
            s1.step(e1, float(e1))
            restored.step(e2, float(e2))

    def test_bo_rng_preserved_after_serialization(self) -> None:
        s1 = self.make_scheduler(seed=42)
        for _ in range(self.WARMUP + 3):
            eps = s1.get_epsilon()
            s1.step(eps, float(eps))

        restored = PLDPBOScheduler.from_state(copy.deepcopy(s1.get_state()))
        gp1_pred = s1._gp.predict(np.array([[0.5], [2.5], [4.5]]), return_std=True)
        gp2_pred = restored._gp.predict(np.array([[0.5], [2.5], [4.5]]), return_std=True)
        np.testing.assert_array_almost_equal(gp1_pred[0], gp2_pred[0])
        np.testing.assert_array_almost_equal(gp1_pred[1], gp2_pred[1])

    # --- Multiple steps ---

    def test_transitions_to_bo_after_enough_steps(self) -> None:
        scheduler = self.make_scheduler(warmup_rounds=3)
        assert scheduler._phase == "warmup"
        scheduler.step(0.1, 0.5)
        scheduler.step(2.5, 0.3)
        scheduler.step(5.0, 0.1)
        assert scheduler._phase == "bo"

    def test_step_forwards_metric_to_gp(self) -> None:
        scheduler = self.make_scheduler(warmup_rounds=3)
        for i in range(3):
            scheduler.step(float(i) + 0.1, float(i))
        assert scheduler._gp is not None
        assert scheduler._f_best == 0.0

    # --- Degenerate cases ---

    def test_degenerate_ei_selects_epsilon_min(self) -> None:
        scheduler = self.make_scheduler(
            warmup_rounds=5,
            acquisition_penalty=1.0,
        )
        for _ in range(5):
            eps = scheduler.get_epsilon()
            scheduler.step(eps, 1.0)
        eps = scheduler.get_epsilon()
        assert eps == pytest.approx(self.EPS_MIN)

    # --- Seed reproducibility ---

    def test_seed_reproducibility(self) -> None:
        s1 = self.make_scheduler(seed=42)
        s2 = self.make_scheduler(seed=42)
        for _ in range(self.WARMUP + 5):
            e1 = s1.get_epsilon()
            e2 = s2.get_epsilon()
            assert e1 == pytest.approx(e2)
            s1.step(e1, 1.0)
            s2.step(e2, 1.0)

    def test_seed_produces_variety(self) -> None:
        s1 = self.make_scheduler(seed=42)
        s2 = self.make_scheduler(seed=99)
        vals1 = []
        vals2 = []
        for _ in range(self.WARMUP + 5):
            e1 = s1.get_epsilon()
            e2 = s2.get_epsilon()
            vals1.append(e1)
            vals2.append(e2)
            s1.step(e1, 1.0)
            s2.step(e2, 1.0)
        assert len(set(vals1) ^ set(vals2)) > 0 or np.std(vals1) > 1e-6

    # --- Validation ---

    def test_invalid_epsilon_min_raises(self) -> None:
        with pytest.raises(ValueError, match="epsilon_min must be positive"):
            PLDPBOScheduler(epsilon_min=0.0, epsilon_max=5.0)

    def test_invalid_epsilon_max_raises(self) -> None:
        with pytest.raises(ValueError, match="epsilon_max must be greater"):
            PLDPBOScheduler(epsilon_min=5.0, epsilon_max=5.0)

    def test_invalid_warmup_rounds_raises(self) -> None:
        with pytest.raises(ValueError, match="warmup_rounds must be at least 2"):
            PLDPBOScheduler(epsilon_min=0.1, epsilon_max=5.0, warmup_rounds=1)

    def test_invalid_acquisition_penalty_raises(self) -> None:
        with pytest.raises(ValueError, match="acquisition_penalty must be non-negative"):
            PLDPBOScheduler(epsilon_min=0.1, epsilon_max=5.0, acquisition_penalty=-1.0)

    def test_invalid_grid_points_raises(self) -> None:
        with pytest.raises(ValueError, match="grid_points must be at least 10"):
            PLDPBOScheduler(epsilon_min=0.1, epsilon_max=5.0, grid_points=5)

    # --- Budget-aware selection ---

    def test_bo_respects_remaining_budget(self) -> None:
        scheduler = self.make_scheduler(acquisition_penalty=0.0, warmup_rounds=5)
        for _ in range(5):
            eps = scheduler.get_epsilon()
            scheduler.step(eps, 0.5)
        scheduler.set_remaining_budget(1.0)
        eps = scheduler.get_epsilon()
        assert eps <= 1.0

    def test_bo_remaining_budget_serialization(self) -> None:
        scheduler = self.make_scheduler(acquisition_penalty=0.0, warmup_rounds=5)
        for _ in range(5):
            eps = scheduler.get_epsilon()
            scheduler.step(eps, 0.5)
        scheduler.set_remaining_budget(2.0)
        state = scheduler.get_state()
        restored = PLDPBOScheduler.from_state(state)
        assert restored._remaining_budget == 2.0

    # --- Config integration ---

    def test_config_integration(self) -> None:
        warmup_rounds = 15
        bo_config = BOConfig(
            enabled=True,
            epsilon_min=0.5,
            epsilon_max=8.0,
            acquisition_penalty=0.2,
            grid_points=200,
            gp_kernel="rbf",
            observation_noise=0.05,
        )
        scheduler = PLDPBOScheduler(
            epsilon_min=bo_config.epsilon_min,
            epsilon_max=bo_config.epsilon_max,
            warmup_rounds=warmup_rounds,
            acquisition_penalty=bo_config.acquisition_penalty,
            grid_points=bo_config.grid_points,
            gp_kernel=bo_config.gp_kernel,
            observation_noise=bo_config.observation_noise,
            seed=42,
        )
        assert scheduler._phase == "warmup"
        # Warm-up count is capped at the 10-point fixed grid
        assert len(scheduler._warmup_epsilons) == len(WARMUP_GRID)
        assert scheduler._warmup_rounds == len(WARMUP_GRID)
        assert scheduler._warmup_epsilons[0] == WARMUP_GRID[0]
        assert scheduler._warmup_epsilons[-1] == WARMUP_GRID[-1]


class TestWarmupGrid:
    """The fixed log-spaced warm-up grid must match the paper (spec §9.3)."""

    def test_ten_points(self) -> None:
        assert len(WARMUP_GRID) == 10

    def test_exact_values(self) -> None:
        assert pytest.approx(
            (0.01, 0.0154, 0.0239, 0.0368, 0.0569, 0.0879, 0.1357, 0.2095, 0.3236, 0.4998),
            rel=0.0, abs=1e-12,
        ) == WARMUP_GRID

    def test_sum_nominal(self) -> None:
        assert pytest.approx(1.3995, rel=1e-12) == WARMUP_SUM_NOMINAL
        assert sum(WARMUP_GRID) == pytest.approx(1.3995, rel=1e-12)

    def test_log_spaced_ratio(self) -> None:
        ratio = 50 ** (1 / 9)  # ~1.5444
        for lo, hi in itertools.pairwise(WARMUP_GRID):
            assert hi / lo == pytest.approx(ratio, abs=1e-2)

    def test_first_and_last_bounds(self) -> None:
        assert WARMUP_GRID[0] == 0.01
        assert WARMUP_GRID[-1] == pytest.approx(0.4998, rel=1e-12)
