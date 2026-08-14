"""Tests for RDP-native mode: sigma calibration, budget enforcement, schedulers."""

from __future__ import annotations

import math

import pytest

from src.privacy.accountant import RDPAccountant
from src.privacy.constants import RDP_ALPHAS
from src.privacy.per_update_dp import (
    _sigma_for_rdp_target,
    _sigma_for_rdp_target_dp_sgd,
    calibrate_sigma_rdp,
    calibrate_sigma_rdp_dp_sgd,
    compute_rdp_cost,
    compute_rdp_cost_dp_sgd,
    enforce_rdp_budget,
)

# ---------------------------------------------------------------------------
# Sigma calibration (per_update)
# ---------------------------------------------------------------------------


class TestSigmaForRDPTarget:
    def test_direct_formula(self) -> None:
        alpha, C, rdp_target = 10.0, 1.0, 0.5
        sigma = _sigma_for_rdp_target(rdp_target, alpha, C)
        assert sigma > 0
        # Verify: RDP(alpha) = alpha * C^2 / (2 * sigma^2) == rdp_target
        rdp = compute_rdp_cost(alpha, sigma, C)
        assert rdp == pytest.approx(rdp_target, rel=1e-10)

    def test_larger_rdp_target_gives_smaller_sigma(self) -> None:
        s1 = _sigma_for_rdp_target(0.1, 10.0, 1.0)
        s2 = _sigma_for_rdp_target(1.0, 10.0, 1.0)
        assert s1 > s2

    def test_larger_alpha_gives_larger_sigma(self) -> None:
        s1 = _sigma_for_rdp_target(0.5, 5.0, 1.0)
        s2 = _sigma_for_rdp_target(0.5, 20.0, 1.0)
        assert s2 > s1

    def test_invalid_rdp_target_raises(self) -> None:
        with pytest.raises(ValueError, match="rdp_target must be positive"):
            _sigma_for_rdp_target(0.0, 10.0, 1.0)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be > 1"):
            _sigma_for_rdp_target(0.5, 1.0, 1.0)

    def test_invalid_clipping_norm_raises(self) -> None:
        with pytest.raises(ValueError, match="clipping_norm must be positive"):
            _sigma_for_rdp_target(0.5, 10.0, 0.0)


class TestCalibrateSigmaRDP:
    def test_returns_correct_sigma(self) -> None:
        alpha, C, rdp_target = 10.0, 1.0, 0.5
        sigma = calibrate_sigma_rdp(rdp_target, alpha, C)
        rdp = compute_rdp_cost(alpha, sigma, C)
        assert rdp == pytest.approx(rdp_target, rel=1e-10)

    def test_min_sigma_clamping(self) -> None:
        sigma = calibrate_sigma_rdp(0.5, 10.0, 1.0, min_sigma=100.0)
        assert sigma == 100.0

    def test_no_clamping_when_sigma_above_min(self) -> None:
        sigma = calibrate_sigma_rdp(0.5, 10.0, 1.0, min_sigma=0.001)
        assert sigma > 0.001


# ---------------------------------------------------------------------------
# Sigma calibration (DP-SGD)
# ---------------------------------------------------------------------------


class TestSigmaForRDPTargetDPSGD:
    def test_direct_formula(self) -> None:
        alpha, q, rdp_target = 10.0, 0.01, 0.5
        sigma = _sigma_for_rdp_target_dp_sgd(rdp_target, alpha, q)
        assert sigma > 0
        rdp = compute_rdp_cost_dp_sgd(alpha, sigma, q)
        assert rdp == pytest.approx(rdp_target, rel=1e-10)

    def test_invalid_sampling_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="sampling_rate must be in"):
            _sigma_for_rdp_target_dp_sgd(0.5, 10.0, 0.0)


class TestCalibrateSigmaRDPDPSGD:
    def test_returns_correct_sigma(self) -> None:
        alpha, q, rdp_target = 10.0, 0.01, 0.5
        sigma = calibrate_sigma_rdp_dp_sgd(rdp_target, alpha, q)
        rdp = compute_rdp_cost_dp_sgd(alpha, sigma, q)
        assert rdp == pytest.approx(rdp_target, rel=1e-10)


# ---------------------------------------------------------------------------
# RDP budget enforcement
# ---------------------------------------------------------------------------


class TestEnforceRDPBudget:
    def test_candidate_fits(self) -> None:
        candidate, budget, current = 0.1, 1.0, 0.0
        rdp_cost, sigma = enforce_rdp_budget(
            candidate, current, budget, rdp_min=0.01,
            alpha=10.0, clipping_norm=1.0,
        )
        assert rdp_cost == pytest.approx(candidate)
        assert sigma > 0

    def test_candidate_exceeds_budget_reduces(self) -> None:
        candidate, budget, current = 0.5, 0.3, 0.0
        rdp_cost, sigma = enforce_rdp_budget(
            candidate, current, budget, rdp_min=0.01,
            alpha=10.0, clipping_norm=1.0,
        )
        assert rdp_cost < candidate
        assert rdp_cost * 1 <= budget  # num_steps=1

    def test_budget_exhausted(self) -> None:
        rdp_cost, sigma = enforce_rdp_budget(
            0.1, 1.0, 0.5, rdp_min=0.01,
            alpha=10.0, clipping_norm=1.0,
        )
        assert rdp_cost == -1.0
        assert sigma == 0.0

    def test_negative_candidate(self) -> None:
        rdp_cost, sigma = enforce_rdp_budget(
            -0.1, 0.0, 1.0, rdp_min=0.01,
            alpha=10.0, clipping_norm=1.0,
        )
        assert rdp_cost == -1.0

    def test_zero_remaining(self) -> None:
        rdp_cost, sigma = enforce_rdp_budget(
            0.1, 1.0, 1.0, rdp_min=0.01,
            alpha=10.0, clipping_norm=1.0,
        )
        assert rdp_cost == -1.0

    def test_per_example_mode(self) -> None:
        rdp_cost, sigma = enforce_rdp_budget(
            0.1, 0.0, 1.0, rdp_min=0.01,
            alpha=10.0, clipping_norm=0.01,
            clipping_mode="per_example", num_steps=1,
            sampling_rate=0.01,
        )
        assert rdp_cost == pytest.approx(0.1)
        assert sigma > 0

    def test_num_steps_multiplied(self) -> None:
        # With num_steps=5, the effective cost is 5x the per-step cost
        rdp_cost, sigma = enforce_rdp_budget(
            0.1, 0.0, 0.4, rdp_min=0.01,
            alpha=10.0, clipping_norm=1.0, num_steps=5,
        )
        # 0.1 * 5 = 0.5 > 0.4, so it should be reduced
        assert rdp_cost < 0.1
        assert rdp_cost * 5 <= 0.4


# ---------------------------------------------------------------------------
# Accountant get_rdp_at_alpha
# ---------------------------------------------------------------------------


class TestAccountantGetRDPAtAlpha:
    def test_zero_steps(self) -> None:
        acc = RDPAccountant(delta=1e-5)
        assert acc.get_rdp_at_alpha(10.0) == 0.0

    def test_single_step(self) -> None:
        acc = RDPAccountant(delta=1e-5)
        sigma, C = 2.0, 1.0
        acc.step(sigma=sigma, clipping_norm=C, num_steps=1)
        expected = compute_rdp_cost(10.0, sigma, C)
        assert acc.get_rdp_at_alpha(10.0) == pytest.approx(expected, rel=1e-10)

    def test_multiple_steps_additive(self) -> None:
        acc = RDPAccountant(delta=1e-5)
        sigma, C = 2.0, 1.0
        acc.step(sigma=sigma, clipping_norm=C, num_steps=3)
        expected = 3 * compute_rdp_cost(10.0, sigma, C)
        assert acc.get_rdp_at_alpha(10.0) == pytest.approx(expected, rel=1e-10)

    def test_interpolation_between_alphas(self) -> None:
        acc = RDPAccountant(delta=1e-5)
        sigma, C = 2.0, 1.0
        acc.step(sigma=sigma, clipping_norm=C, num_steps=1)
        # Alpha between two grid points
        alpha_mid = 10.5
        rdp = acc.get_rdp_at_alpha(alpha_mid)
        rdp_lo = acc.get_rdp_at_alpha(10.0)
        rdp_hi = acc.get_rdp_at_alpha(11.0)
        assert rdp_lo <= rdp <= rdp_hi or rdp_hi <= rdp <= rdp_lo

    def test_alpha_below_grid(self) -> None:
        acc = RDPAccountant(delta=1e-5)
        acc.step(sigma=2.0, clipping_norm=1.0, num_steps=1)
        rdp = acc.get_rdp_at_alpha(1.5)
        assert rdp == pytest.approx(acc.get_rdp_at_alpha(float(RDP_ALPHAS[0])), rel=1e-10)

    def test_alpha_above_grid(self) -> None:
        acc = RDPAccountant(delta=1e-5)
        acc.step(sigma=2.0, clipping_norm=1.0, num_steps=1)
        rdp = acc.get_rdp_at_alpha(200.0)
        assert rdp == pytest.approx(acc.get_rdp_at_alpha(float(RDP_ALPHAS[-1])), rel=1e-10)


# ---------------------------------------------------------------------------
# RDP-native schedulers
# ---------------------------------------------------------------------------


class TestFixedRDPScheduler:
    def test_basic(self) -> None:
        from src.privacy.epsilon_scheduler import FixedRDPScheduler
        s = FixedRDPScheduler(rdp_target=0.5)
        assert s.get_rdp() == 0.5

    def test_invalid_target_raises(self) -> None:
        from src.privacy.epsilon_scheduler import FixedRDPScheduler
        with pytest.raises(ValueError, match="rdp_target must be positive"):
            FixedRDPScheduler(rdp_target=0.0)

    def test_serialization(self) -> None:
        from src.privacy.epsilon_scheduler import FixedRDPScheduler
        s = FixedRDPScheduler(rdp_target=0.5)
        state = s.get_state()
        restored = FixedRDPScheduler.from_state(state)
        assert restored.get_rdp() == 0.5

    def test_repr(self) -> None:
        from src.privacy.epsilon_scheduler import FixedRDPScheduler
        s = FixedRDPScheduler(rdp_target=0.5)
        assert "0.5" in repr(s)


class TestUniformRandomRDPScheduler:
    def test_values_in_range(self) -> None:
        from src.privacy.epsilon_scheduler import UniformRandomRDPScheduler
        s = UniformRandomRDPScheduler(rdp_min=0.1, rdp_max=1.0, seed=42)
        for _ in range(50):
            rdp = s.get_rdp()
            assert 0.1 <= rdp <= 1.0

    def test_deterministic_with_seed(self) -> None:
        from src.privacy.epsilon_scheduler import UniformRandomRDPScheduler
        s1 = UniformRandomRDPScheduler(rdp_min=0.1, rdp_max=1.0, seed=42)
        s2 = UniformRandomRDPScheduler(rdp_min=0.1, rdp_max=1.0, seed=42)
        for _ in range(10):
            assert s1.get_rdp() == s2.get_rdp()

    def test_serialization(self) -> None:
        from src.privacy.epsilon_scheduler import UniformRandomRDPScheduler
        s = UniformRandomRDPScheduler(rdp_min=0.1, rdp_max=1.0, seed=42)
        s.get_rdp()  # advance RNG
        state = s.get_state()
        restored = UniformRandomRDPScheduler.from_state(state)
        assert restored.get_rdp() == s.get_rdp()

    def test_invalid_range_raises(self) -> None:
        from src.privacy.epsilon_scheduler import UniformRandomRDPScheduler
        with pytest.raises(ValueError, match="rdp_min must be positive"):
            UniformRandomRDPScheduler(rdp_min=0.0, rdp_max=1.0)


class TestPLDPBORDPScheduler:
    def test_warmup(self) -> None:
        from src.privacy.bo_scheduler import WARMUP_GRID, PLDPBORDPScheduler
        s = PLDPBORDPScheduler(
            rdp_min=0.1, rdp_max=1.0, warmup_rounds=5, seed=42,
        )
        values = []
        for _ in range(5):
            rdp = s.get_rdp()
            values.append(rdp)
            s.step(rdp, 1.0)
        # Fixed log-spaced grid (spec §9.3): the first 5 points, regardless
        # of the BO search bounds
        assert values == pytest.approx(list(WARMUP_GRID[:5]), rel=0.0, abs=1e-12)
        assert values[0] == pytest.approx(WARMUP_GRID[0])
        assert values[-1] == pytest.approx(WARMUP_GRID[4])

    def test_bo_phase(self) -> None:
        from src.privacy.bo_scheduler import PLDPBORDPScheduler
        s = PLDPBORDPScheduler(
            rdp_min=0.1, rdp_max=1.0, warmup_rounds=3, seed=42,
        )
        for _ in range(3):
            rdp = s.get_rdp()
            s.step(rdp, 1.0)
        # After warmup, should be in BO phase
        rdp = s.get_rdp()
        assert 0.1 <= rdp <= 1.0

    def test_serialization(self) -> None:
        from src.privacy.bo_scheduler import PLDPBORDPScheduler
        s = PLDPBORDPScheduler(
            rdp_min=0.1, rdp_max=1.0, warmup_rounds=3, seed=42,
        )
        for _ in range(5):
            rdp = s.get_rdp()
            s.step(rdp, 1.0)
        state = s.get_state()
        restored = PLDPBORDPScheduler.from_state(state)
        assert restored.get_rdp() == s.get_rdp()
        assert restored._round == s._round

    def test_remaining_budget(self) -> None:
        from src.privacy.bo_scheduler import PLDPBORDPScheduler
        s = PLDPBORDPScheduler(
            rdp_min=0.1, rdp_max=1.0, warmup_rounds=3, seed=42,
        )
        s.set_remaining_budget(0.2)
        # After warmup, values should be masked to <= 0.2
        for _ in range(3):
            rdp = s.get_rdp()
            s.step(rdp, 1.0)
        rdp = s.get_rdp()
        assert rdp <= 0.2

    def test_invalid_params_raise(self) -> None:
        from src.privacy.bo_scheduler import PLDPBORDPScheduler
        with pytest.raises(ValueError, match="rdp_min must be positive"):
            PLDPBORDPScheduler(rdp_min=0.0, rdp_max=1.0)
        with pytest.raises(ValueError, match="rdp_max must be greater"):
            PLDPBORDPScheduler(rdp_min=1.0, rdp_max=1.0)


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_rdp_native_config_fields(self) -> None:
        from src.config.loader import BOConfig, PrivacyConfig
        pc = PrivacyConfig(accountant_mode="rdp_native", rdp_alpha=12.0)
        assert pc.accountant_mode == "rdp_native"
        assert pc.rdp_alpha == 12.0

        bc = BOConfig(rdp_min=0.05, rdp_max=1.5)
        assert bc.rdp_min == 0.05
        assert bc.rdp_max == 1.5

    def test_default_config_backward_compatible(self) -> None:
        from src.config.loader import BOConfig, PrivacyConfig
        pc = PrivacyConfig()
        assert pc.accountant_mode == "epsilon"
        assert pc.rdp_alpha == 10.0

        bc = BOConfig()
        assert bc.rdp_min == 0.01
        assert bc.rdp_max == 2.0


# ---------------------------------------------------------------------------
# Per-round RDP accounting (spec §2): 1 round = 1 Gaussian release
# ---------------------------------------------------------------------------


class TestPerRoundRDPCalibration:
    @pytest.mark.parametrize("r_t,q", [(0.5, 0.1), (0.01, 0.05), (1.0, 0.5)])
    def test_sigma_hits_per_round_target(self, r_t: float, q: float) -> None:
        alpha = 10.0
        sigma = _sigma_for_rdp_target_dp_sgd(r_t, alpha, q)
        # sigma_t = sqrt(alpha * q^2 / (2 * R_t)) — the paper's closed form
        assert sigma == pytest.approx(
            math.sqrt(alpha * q**2 / (2.0 * r_t)), rel=1e-12,
        )
        rdp = compute_rdp_cost_dp_sgd(alpha, sigma, q)
        assert rdp == pytest.approx(r_t, rel=1e-9)

    def test_accountant_round_step_cost_matches_target(self) -> None:
        alpha, q, r_t = 10.0, 0.064, 0.5
        sigma = _sigma_for_rdp_target_dp_sgd(r_t, alpha, q)
        accountant = RDPAccountant(delta=1e-5)
        accountant.step(
            sigma=sigma, clipping_norm=q, num_steps=1, mode="per_example",
        )
        cost = accountant.get_rdp_at_alpha(alpha)
        assert cost == pytest.approx(r_t, rel=1e-6)


class TestResolveRDPerRound:
    """_resolve_rdp per_example branch: candidate/sigma stay per-round."""

    @staticmethod
    def _make_config():
        from src.config.loader import ExperimentConfig
        cfg = ExperimentConfig()
        cfg.privacy.enabled = True
        cfg.privacy.accountant_mode = "rdp_native"
        cfg.privacy.clipping_mode = "per_example"
        cfg.privacy.rdp_alpha = 10.0
        cfg.privacy.update_clip_norm = 1.0
        cfg.data.batch_size = 64
        return cfg

    @staticmethod
    def _expected_sigma(r_t: float, q: float, alpha: float = 10.0) -> float:
        return math.sqrt(alpha * q**2 / (2.0 * r_t))

    def test_candidate_fits_uses_per_round_sigma(self) -> None:
        from src.client_app import _resolve_rdp
        from src.privacy.epsilon_scheduler import FixedRDPScheduler
        config = self._make_config()
        accountant = RDPAccountant(delta=1e-5)
        q = 64 / 1000
        r_t = 0.5
        rdp_cost, sigma = _resolve_rdp(
            FixedRDPScheduler(rdp_target=r_t), accountant, config,
            total_budget=10.0, eps_min=0.01, local_train_size=1000,
        )
        assert rdp_cost == pytest.approx(r_t, rel=1e-9)
        assert sigma == pytest.approx(self._expected_sigma(r_t, q), rel=1e-9)

    def test_candidate_reduced_to_fit_budget(self) -> None:
        from src.client_app import _resolve_rdp
        from src.privacy.epsilon_scheduler import FixedRDPScheduler
        config = self._make_config()
        alpha = config.privacy.rdp_alpha
        q = 64 / 1000
        current = 0.9
        accountant = RDPAccountant(delta=1e-5)
        sigma_current = _sigma_for_rdp_target_dp_sgd(current, alpha, q)
        accountant.step(
            sigma=sigma_current, clipping_norm=q, num_steps=1,
            mode="per_example",
        )
        rdp_cost, sigma = _resolve_rdp(
            FixedRDPScheduler(rdp_target=0.5), accountant, config,
            total_budget=1.0, eps_min=0.01, local_train_size=1000,
        )
        assert 0.01 <= rdp_cost < 0.5
        assert current + rdp_cost <= 1.0 + 1e-9
        # sigma must be consistent with the enforced per-round cost
        assert compute_rdp_cost_dp_sgd(alpha, sigma, q) == pytest.approx(
            rdp_cost, rel=1e-9,
        )

    def test_budget_exhausted_returns_minus_one(self) -> None:
        from src.client_app import _resolve_rdp
        from src.privacy.epsilon_scheduler import FixedRDPScheduler
        config = self._make_config()
        alpha = config.privacy.rdp_alpha
        q = 64 / 1000
        accountant = RDPAccountant(delta=1e-5)
        sigma_used = _sigma_for_rdp_target_dp_sgd(0.995, alpha, q)
        accountant.step(
            sigma=sigma_used, clipping_norm=q, num_steps=1, mode="per_example",
        )
        rdp_cost, sigma = _resolve_rdp(
            FixedRDPScheduler(rdp_target=0.5), accountant, config,
            total_budget=1.0, eps_min=0.01, local_train_size=1000,
        )
        assert rdp_cost == -1.0
        assert sigma == 0.0

    def test_no_budget_direct_calibration(self) -> None:
        from src.client_app import _resolve_rdp
        from src.privacy.epsilon_scheduler import FixedRDPScheduler
        config = self._make_config()
        q = 64 / 1000
        r_t = 0.5
        rdp_cost, sigma = _resolve_rdp(
            FixedRDPScheduler(rdp_target=r_t), None, config,
            local_train_size=1000,
        )
        assert rdp_cost == pytest.approx(r_t, rel=1e-9)
        assert sigma == pytest.approx(self._expected_sigma(r_t, q), rel=1e-9)


class TestPerExampleClientRoundParity:
    """Fit-level parity: acct_cost == r_t_final == accountant cumulative."""

    def test_fit_reports_round_parity(self) -> None:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from src.client.per_example_dp_client import PerExampleDPClient
        from src.config.loader import ExperimentConfig
        from src.models.base import BaseModel

        class _TinyModel(BaseModel):
            def __init__(self) -> None:
                self._net = nn.Linear(10, 2)

            def get_model(self) -> nn.Module:
                return self._net

        config = ExperimentConfig()
        config.privacy.enabled = True
        config.privacy.accountant_mode = "rdp_native"
        config.privacy.clipping_mode = "per_example"
        config.privacy.rdp_alpha = 10.0
        config.privacy.update_clip_norm = 1.0
        config.optimizer.momentum = 0.0
        config.data.batch_size = 2

        data = TensorDataset(torch.randn(4, 10), torch.randint(0, 2, (4,)))
        loader = DataLoader(data, batch_size=2)
        alpha, q, r_t = 10.0, 2 / 4, 0.5
        sigma = _sigma_for_rdp_target_dp_sgd(r_t, alpha, q)

        client = PerExampleDPClient(
            _TinyModel(), loader, loader, config,
            client_epsilon=r_t, computed_sigma=sigma,
            accountant=RDPAccountant(delta=1e-5),
        )
        weights, num_examples, metrics = client.fit(
            client.get_parameters({}), {},
        )
        assert isinstance(weights, list)
        assert num_examples == 4
        assert metrics["r_t_final"] == pytest.approx(r_t, rel=1e-6)
        assert metrics["rdp_cost"] == pytest.approx(r_t, rel=1e-6)
        assert metrics["acct_cost"] == pytest.approx(
            metrics["r_t_final"], rel=1e-6,
        )
        assert metrics["cumulative_rdp"] == pytest.approx(
            metrics["acct_cost"], rel=1e-6,
        )
