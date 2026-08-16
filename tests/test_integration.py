from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.config.loader import load_config
from src.privacy.accountant import RDPAccountant
from src.privacy.bo_scheduler import WARMUP_GRID, PLDPBOScheduler
from src.privacy.epsilon_scheduler import (
    FixedEpsilonScheduler,
    UniformRandomEpsilonScheduler,
)
from src.privacy.per_update_dp import (
    calibrate_sigma,
    enforce_epsilon_budget,
)

_OPTIMIZATION_METRIC_KEY_MAP: dict[str, str] = {
    "nun": "update_norm",
    "utility": "utility_loss",
}


class TestConfigLoading:
    def test_pldp_bo_nun_config_loads(self) -> None:
        config = load_config("config/experiments/archive/pldp_bo_mnist_iid_logit_disagreement.yaml")
        assert config.federated.strategy == "pldp_bo"
        assert config.bo.enabled
        assert config.bo.optimization_metric == "logit_disagreement"
        assert config.personalization.enabled
        assert config.personalization.strategy == "data_proportional"
        assert config.federated.server_learning_rate == 0.5
        assert config.data.name == "mnist"
        assert config.model.name == "mlp"

    def test_pldp_bo_utility_config_loads(self) -> None:
        config = load_config("config/experiments/archive/pldp_bo_mnist_iid_utility_retention.yaml")
        assert config.federated.strategy == "pldp_bo"
        assert config.bo.enabled
        assert config.bo.optimization_metric == "utility_retention"

    def test_pldp_bo_noniid_config_loads(self) -> None:
        config = load_config(
            "config/experiments/archive/pldp_bo_cifar100_noniid_logit_disagreement.yaml",
        )
        assert config.data.partition_type == "noniid"
        assert config.data.partition_alpha == 0.5
        assert config.model.name == "cnn"
        assert config.model.num_classes == 100

    def test_fedavg_dp_config_loads(self) -> None:
        config = load_config("config/experiments/archive/fedavg_mnist_iid.yaml")
        assert config.federated.strategy == "fedavg"
        assert not config.bo.enabled
        assert config.personalization.enabled

    def test_all_pldp_bo_configs_load(self) -> None:
        paths = [
            "config/experiments/archive/pldp_bo_mnist_iid_logit_disagreement.yaml",
            "config/experiments/archive/pldp_bo_mnist_noniid_logit_disagreement.yaml",
            "config/experiments/archive/pldp_bo_cifar100_iid_logit_disagreement.yaml",
            "config/experiments/archive/pldp_bo_cifar100_noniid_logit_disagreement.yaml",
            "config/experiments/archive/pldp_bo_mnist_iid_utility_retention.yaml",
            "config/experiments/archive/pldp_bo_mnist_noniid_utility_retention.yaml",
            "config/experiments/archive/pldp_bo_cifar100_iid_utility_retention.yaml",
            "config/experiments/archive/pldp_bo_cifar100_noniid_utility_retention.yaml",
        ]
        for path in paths:
            config = load_config(path)
            assert config.federated.strategy == "pldp_bo"
            assert config.bo.enabled
            assert config.privacy.enabled


class TestFullRoundLifecycle:
    """End-to-end test of the per-round client lifecycle with PLDP-BO."""

    # Aligned to the fixed log-spaced warm-up grid (spec §9.3): the scheduler's
    # warm-up candidates live in [WARMUP_GRID[0], WARMUP_GRID[-1]].
    EPS_MIN = WARMUP_GRID[0]
    EPS_MAX = WARMUP_GRID[-1]
    WARMUP = 3
    TOTAL_ROUNDS = 5
    C = 1.0
    DELTA = 1e-5
    BUDGET = 8.0

    @pytest.fixture
    def config(self) -> object:
        return load_config("config/experiments/archive/pldp_bo_mnist_iid_logit_disagreement.yaml")

    @pytest.fixture
    def accountant(self) -> RDPAccountant:
        return RDPAccountant(delta=self.DELTA)

    @pytest.fixture
    def scheduler(self) -> PLDPBOScheduler:
        return PLDPBOScheduler(
            epsilon_min=self.EPS_MIN,
            epsilon_max=self.EPS_MAX,
            warmup_rounds=self.WARMUP,
            seed=42,
        )

    def _resolve_epsilon(
        self,
        candidate: float,
        accountant: RDPAccountant,
    ) -> tuple[float, float]:
        return enforce_epsilon_budget(
            candidate,
            accountant.rdp_per_alpha,
            self.BUDGET,
            self.EPS_MIN,
            self.C,
            self.DELTA,
        )

    def _simulate_training_metric(self, epsilon: float) -> float:
        return 1.0 / (1.0 + epsilon)

    def _resolve_and_check(
        self,
        candidate: float,
        accountant: RDPAccountant,
    ) -> float | None:
        """Return resolved epsilon, or None if budget is exhausted."""
        epsilon, _ = self._resolve_epsilon(candidate, accountant)
        if epsilon < 0:
            return None
        assert epsilon <= candidate + 1e-12
        assert self.EPS_MIN <= epsilon <= self.EPS_MAX
        return epsilon

    def _run_round(
        self,
        scheduler: PLDPBOScheduler,
        accountant: RDPAccountant,
    ) -> bool:
        """Run one training round. Return False if budget exhausted."""
        candidate = scheduler.get_epsilon()
        assert self.EPS_MIN <= candidate <= self.EPS_MAX
        epsilon = self._resolve_and_check(candidate, accountant)
        if epsilon is None:
            return False
        sigma = calibrate_sigma(epsilon, self.C, self.DELTA)
        accountant.step(sigma=sigma, clipping_norm=self.C, num_steps=1)
        metric = self._simulate_training_metric(epsilon)
        scheduler.step(epsilon, metric)
        return True

    def test_full_warmup_phase(
        self,
        accountant: RDPAccountant,
        scheduler: PLDPBOScheduler,
    ) -> None:
        for _ in range(self.WARMUP):
            candidate = scheduler.get_epsilon()
            assert self.EPS_MIN <= candidate <= self.EPS_MAX

            epsilon, computed_sigma = self._resolve_epsilon(candidate, accountant)
            assert epsilon <= candidate + 1e-12
            assert self.EPS_MIN <= epsilon <= self.EPS_MAX

            sigma = (
                computed_sigma
                if computed_sigma > 0
                else calibrate_sigma(epsilon, self.C, self.DELTA)
            )
            accountant.step(sigma=sigma, clipping_norm=self.C, num_steps=1)

            metric = self._simulate_training_metric(epsilon)
            scheduler.step(epsilon, metric)

        assert scheduler._phase == "bo"

    def test_full_bo_phase(
        self,
        accountant: RDPAccountant,
        scheduler: PLDPBOScheduler,
    ) -> None:
        for _ in range(self.WARMUP):
            if not self._run_round(scheduler, accountant):
                break

        assert scheduler._phase == "bo"

        for _ in range(self.WARMUP, self.TOTAL_ROUNDS):
            if not self._run_round(scheduler, accountant):
                break

        assert scheduler._phase == "bo"

    def test_state_persistence_across_rounds(
        self,
        accountant: RDPAccountant,
        scheduler: PLDPBOScheduler,
    ) -> None:
        acct_states = []
        sched_states = []

        for _ in range(self.WARMUP + 2):
            candidate = scheduler.get_epsilon()
            epsilon, computed_sigma = self._resolve_epsilon(candidate, accountant)
            if epsilon < 0:
                break
            sigma = (
                computed_sigma
                if computed_sigma > 0
                else calibrate_sigma(epsilon, self.C, self.DELTA)
            )
            accountant.step(sigma=sigma, clipping_norm=self.C, num_steps=1)
            metric = self._simulate_training_metric(epsilon)
            scheduler.step(epsilon, metric)

            acct_states.append(accountant.get_state())
            sched_states.append(scheduler.get_state())

            restored_acct = RDPAccountant.from_state(copy.deepcopy(acct_states[-1]))
            restored_sched = PLDPBOScheduler.from_state(copy.deepcopy(sched_states[-1]))

            assert restored_acct.get_epsilon() == pytest.approx(accountant.get_epsilon())
            assert restored_sched._phase == scheduler._phase
            assert restored_sched._round == scheduler._round
            assert restored_sched._observations == scheduler._observations

    def test_budget_enforcement_during_bo(
        self,
        accountant: RDPAccountant,
        scheduler: PLDPBOScheduler,
    ) -> None:
        for _ in range(self.WARMUP):
            if not self._run_round(scheduler, accountant):
                break

        for _ in range(self.WARMUP, 20):
            if not self._run_round(scheduler, accountant):
                break

        cum_eps = accountant.get_epsilon()
        assert cum_eps <= self.BUDGET * 1.05, (
            f"Cumulative epsilon {cum_eps:.4f} exceeds budget {self.BUDGET:.4f}"
        )

    def test_fixed_scheduler_lifecycle(self) -> None:
        accountant = RDPAccountant(delta=self.DELTA)
        scheduler = FixedEpsilonScheduler(epsilon=2.0)

        for _ in range(5):
            candidate = scheduler.get_epsilon()
            assert candidate == 2.0
            if accountant is not None:
                epsilon, computed_sigma = self._resolve_epsilon(candidate, accountant)
            else:
                epsilon = candidate
                computed_sigma = 0.0
            sigma = (
                computed_sigma
                if computed_sigma > 0
                else calibrate_sigma(epsilon, self.C, self.DELTA)
            )
            accountant.step(sigma=sigma, clipping_norm=self.C, num_steps=1)

        assert accountant.get_epsilon() > 0

    def test_uniform_random_scheduler_lifecycle(self) -> None:
        accountant = RDPAccountant(delta=self.DELTA)
        scheduler = UniformRandomEpsilonScheduler(
            epsilon_min=self.EPS_MIN,
            epsilon_max=self.EPS_MAX,
            seed=42,
        )

        for _ in range(10):
            candidate = scheduler.get_epsilon()
            assert self.EPS_MIN <= candidate <= self.EPS_MAX
            if accountant is not None:
                epsilon, computed_sigma = self._resolve_epsilon(candidate, accountant)
                if epsilon < 0:
                    break
            else:
                epsilon = candidate
                computed_sigma = 0.0
            sigma = (
                computed_sigma
                if computed_sigma > 0
                else calibrate_sigma(epsilon, self.C, self.DELTA)
            )
            accountant.step(sigma=sigma, clipping_norm=self.C, num_steps=1)

        assert accountant.get_epsilon() > 0

    def test_multiple_clients_independent_bo(self) -> None:
        accountants = [RDPAccountant(delta=self.DELTA) for _ in range(3)]
        schedulers = [
            PLDPBOScheduler(
                epsilon_min=self.EPS_MIN,
                epsilon_max=self.EPS_MAX,
                warmup_rounds=self.WARMUP,
                seed=seed,
            )
            for seed in [42, 99, 123]
        ]

        # Real clients see different data, hence different metric functions;
        # identical environments correctly yield identical schedules.
        def metric(client_idx: int, epsilon: float) -> float:
            if client_idx == 0:
                return 1.0 / (1.0 + epsilon)
            if client_idx == 1:
                return 0.5 + 0.01 * epsilon
            return 0.5

        for _ in range(self.WARMUP + 2):
            for idx, (acct, sched) in enumerate(zip(accountants, schedulers, strict=True)):
                candidate = sched.get_epsilon()
                epsilon, computed_sigma = self._resolve_epsilon(candidate, acct)
                if epsilon < 0:
                    continue
                sigma = (
                    computed_sigma
                    if computed_sigma > 0
                    else calibrate_sigma(epsilon, self.C, self.DELTA)
                )
                acct.step(sigma=sigma, clipping_norm=self.C, num_steps=1)
                sched.step(epsilon, metric(idx, epsilon))

        epsilons = [acct.get_epsilon() for acct in accountants]
        assert all(eps > 0 for eps in epsilons)
        assert not all(eps == epsilons[0] for eps in epsilons)

    def test_optimization_metric_key_mapping(self) -> None:
        assert _OPTIMIZATION_METRIC_KEY_MAP["nun"] == "update_norm"
        assert _OPTIMIZATION_METRIC_KEY_MAP["utility"] == "utility_loss"

    def test_scheduler_receives_correct_metric(self) -> None:
        scheduler = PLDPBOScheduler(
            epsilon_min=self.EPS_MIN,
            epsilon_max=self.EPS_MAX,
            warmup_rounds=self.WARMUP,
            seed=42,
        )
        accountant = RDPAccountant(delta=self.DELTA)

        # Simulate a round with NUN metric
        epsilon = scheduler.get_epsilon()
        sigma = calibrate_sigma(epsilon, self.C, self.DELTA)
        accountant.step(sigma=sigma, clipping_norm=self.C, num_steps=1)

        for metric_config, _metrics_key, value in [
            ("nun", "update_norm", 1.5),
            ("utility", "utility_loss", 0.75),
        ]:
            scheduler_copy = PLDPBOScheduler(
                epsilon_min=self.EPS_MIN,
                epsilon_max=self.EPS_MAX,
                warmup_rounds=self.WARMUP,
                seed=42,
            )
            # Step through warmup to get to BO
            for _ in range(self.WARMUP):
                eps = scheduler_copy.get_epsilon()
                scheduler_copy.step(eps, 0.5)

            _OPTIMIZATION_METRIC_KEY_MAP[metric_config]
            scheduler_copy.step(epsilon, value)

            assert scheduler_copy._observations[-1][1] == pytest.approx(value)

    def test_budget_exhausted_does_not_step_scheduler(self) -> None:
        scheduler = PLDPBOScheduler(
            epsilon_min=self.EPS_MIN,
            epsilon_max=self.EPS_MAX,
            warmup_rounds=self.WARMUP,
            seed=42,
        )
        accountant = RDPAccountant(delta=self.DELTA)

        # Run warmup to get beyond phase transition
        for _ in range(self.WARMUP + 1):
            eps = scheduler.get_epsilon()
            sigma = calibrate_sigma(eps, self.C, self.DELTA)
            accountant.step(sigma=sigma, clipping_norm=self.C, num_steps=1)
            scheduler.step(eps, 0.5)

        round_before = scheduler._round
        n_obs_before = len(scheduler._observations)

        # Simulate the guard in client_app.py: when budget_exhausted is True,
        # scheduler.step() should NOT be called
        epsilon = scheduler.get_epsilon()
        fit_metrics = {"budget_exhausted": True, "update_norm": 1.5, "utility_loss": 0.5}
        if not fit_metrics.get("budget_exhausted", False):
            metric_key = _OPTIMIZATION_METRIC_KEY_MAP["nun"]
            metric_value = fit_metrics.get(metric_key)
            if metric_value is not None:
                scheduler.step(epsilon, float(metric_value))

        assert scheduler._round == round_before, "scheduler._round should not increment"
        assert len(scheduler._observations) == n_obs_before, "should not add observation"


class TestFixedBaselineBudgetMatch:
    """IMPL-09 §9.5: FixedRDPScheduler(0.5) fills B_RDP=10.0 in exactly
    20 participations; the 21st is refused (cumulative 10.0 + 0.5 > 10.0)."""

    BUDGET = 10.0
    CANDIDATE = 0.5
    ALPHA = 10.0

    def test_accepts_20_participations_then_refuses(self) -> None:
        from src.privacy.epsilon_scheduler import FixedRDPScheduler
        from src.privacy.per_update_dp import enforce_rdp_budget

        scheduler = FixedRDPScheduler(rdp_target=self.CANDIDATE)
        assert scheduler.get_rdp() == pytest.approx(self.CANDIDATE)
        assert getattr(scheduler, "_phase", None) is None

        current = 0.0
        accepted = 0
        for _ in range(40):
            rdp_cost, sigma = enforce_rdp_budget(
                scheduler.get_rdp(),
                current,
                self.BUDGET,
                1e-6,
                self.ALPHA,
                1.0,
                clipping_mode="per_update",
                num_steps=1,
            )
            if rdp_cost < 0:
                break
            assert rdp_cost == pytest.approx(self.CANDIDATE)
            assert sigma > 0
            current += rdp_cost
            accepted += 1

        assert accepted == 20
        assert current == pytest.approx(self.BUDGET)
        rdp_cost, _ = enforce_rdp_budget(
            self.CANDIDATE,
            current,
            self.BUDGET,
            1e-6,
            self.ALPHA,
            1.0,
            clipping_mode="per_update",
            num_steps=1,
        )
        assert rdp_cost < 0


class TestFourCellMatrixSmoke:
    """IMPL-09 §9.5: the 4-cell method/aggregation matrix wires to the right
    client scheduler and server strategy."""

    CELLS = [
        ("nonprivate", "plain", None, "SafeFedAvg"),
        ("dpfedavg_fixed", "attenuation", "FixedRDPScheduler", "MedianRobustAggregation"),
        ("fedprox_fixed", "attenuation", "FixedRDPScheduler", "MedianRobustAggregation"),
        ("pldpbo_nun", "attenuation", "PLDPBORDPScheduler", "MedianRobustAggregation"),
    ]
    METHOD_OVERRIDES: dict[str, dict[str, bool | float]] = {
        "nonprivate": {"privacy.enabled": False, "bo.enabled": False},
        "dpfedavg_fixed": {"privacy.enabled": True, "bo.enabled": False},
        "fedprox_fixed": {
            "privacy.enabled": True,
            "bo.enabled": False,
            "federated.proximal_mu": 0.01,
        },
        "pldpbo_nun": {"privacy.enabled": True, "bo.enabled": True},
    }

    @pytest.mark.parametrize(("method", "aggregation", "scheduler_cls", "strategy_cls"), CELLS)
    def test_cell_wiring(
        self,
        method: str,
        aggregation: str,
        scheduler_cls: str | None,
        strategy_cls: str,  # noqa: ARG002
    ) -> None:
        from src.config.locked import collect_violations
        from src.server.strategy import MedianRobustAggregation, SafeFedAvg
        from src.server_app import _make_strategy

        cfg = load_config(
            "config/default.yaml",
            overrides={
                "method": method,
                "federated.aggregation": aggregation,
                "assert_locked_config": False,
                **self.METHOD_OVERRIDES[method],
            },
        )

        assert all(not v.startswith("method") for v in collect_violations(cfg)), collect_violations(
            cfg
        )

        if method == "nonprivate":
            from src.client_app import _make_scheduler

            assert _make_scheduler(0, object(), cfg, 1) is None
            assert isinstance(_make_strategy(cfg, None, None, None), SafeFedAvg)
        else:
            from src.client_app import _make_rdp_native_scheduler
            from src.privacy.bo_scheduler import PLDPBORDPScheduler
            from src.privacy.epsilon_scheduler import FixedRDPScheduler

            scheduler = _make_rdp_native_scheduler(0, cfg)
            expected_cls = (
                FixedRDPScheduler if scheduler_cls == "FixedRDPScheduler" else PLDPBORDPScheduler
            )
            assert isinstance(scheduler, expected_cls)
            assert isinstance(_make_strategy(cfg, None, None, None), MedianRobustAggregation)


class TestFemnistDataLessLoader:
    """IMPL-14: FEMNIST loader fails cleanly without processed data."""

    def test_missing_processed_dir_raises_clean_error(self, tmp_path: Path) -> None:
        from src.data.femnist import FEMNISTDataset

        empty_root = tmp_path / "no_femnist"
        empty_root.mkdir()
        with pytest.raises(FileNotFoundError, match="FEMNIST/processed"):
            FEMNISTDataset(root=str(empty_root), train=True, transform=None)
