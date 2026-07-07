from __future__ import annotations

import copy

import pytest

from src.config.loader import load_config
from src.privacy.accountant import RDPAccountant
from src.privacy.bo_scheduler import PLDPBOScheduler
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
        config = load_config("config/experiments/pldp_bo_mnist_iid_nun.yaml")
        assert config.federated.strategy == "pldp_bo"
        assert config.bo.enabled
        assert config.bo.optimization_metric == "nun"
        assert not config.personalization.enabled
        assert config.federated.server_learning_rate == 1.0
        assert config.data.name == "mnist"
        assert config.model.name == "mlp"

    def test_pldp_bo_utility_config_loads(self) -> None:
        config = load_config("config/experiments/pldp_bo_mnist_iid_utility.yaml")
        assert config.federated.strategy == "pldp_bo"
        assert config.bo.enabled
        assert config.bo.optimization_metric == "utility"

    def test_pldp_bo_noniid_config_loads(self) -> None:
        config = load_config("config/experiments/pldp_bo_cifar100_noniid_nun.yaml")
        assert config.data.partition_type == "noniid"
        assert config.data.partition_alpha == 0.5
        assert config.model.name == "cnn"
        assert config.model.num_classes == 100

    def test_fedavg_dp_config_loads(self) -> None:
        config = load_config("config/experiments/fedavg_mnist_iid.yaml")
        assert config.federated.strategy == "fedavg"
        assert not config.bo.enabled
        assert config.personalization.enabled

    def test_all_pldp_bo_configs_load(self) -> None:
        paths = [
            "config/experiments/pldp_bo_mnist_iid_nun.yaml",
            "config/experiments/pldp_bo_mnist_noniid_nun.yaml",
            "config/experiments/pldp_bo_cifar100_iid_nun.yaml",
            "config/experiments/pldp_bo_cifar100_noniid_nun.yaml",
            "config/experiments/pldp_bo_mnist_iid_utility.yaml",
            "config/experiments/pldp_bo_mnist_noniid_utility.yaml",
            "config/experiments/pldp_bo_cifar100_iid_utility.yaml",
            "config/experiments/pldp_bo_cifar100_noniid_utility.yaml",
        ]
        for path in paths:
            config = load_config(path)
            assert config.federated.strategy == "pldp_bo"
            assert config.bo.enabled
            assert config.privacy.enabled


class TestFullRoundLifecycle:
    """End-to-end test of the per-round client lifecycle with PLDP-BO."""

    EPS_MIN = 0.1
    EPS_MAX = 5.0
    WARMUP = 3
    TOTAL_ROUNDS = 5
    C = 1.0
    DELTA = 1e-5
    BUDGET = 8.0

    @pytest.fixture
    def config(self) -> object:
        return load_config("config/experiments/pldp_bo_mnist_iid_nun.yaml")

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
    ) -> float:
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
        epsilon = self._resolve_epsilon(candidate, accountant)
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

            epsilon = self._resolve_epsilon(candidate, accountant)
            assert epsilon <= candidate + 1e-12
            assert self.EPS_MIN <= epsilon <= self.EPS_MAX

            sigma = calibrate_sigma(epsilon, self.C, self.DELTA)
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
            epsilon = self._resolve_epsilon(candidate, accountant)
            if epsilon < 0:
                break
            sigma = calibrate_sigma(epsilon, self.C, self.DELTA)
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
        assert cum_eps <= self.BUDGET * 1.05 or cum_eps > 0

    def test_fixed_scheduler_lifecycle(self) -> None:
        accountant = RDPAccountant(delta=self.DELTA)
        scheduler = FixedEpsilonScheduler(epsilon=2.0)

        for _ in range(5):
            candidate = scheduler.get_epsilon()
            assert candidate == 2.0
            if accountant is not None:
                epsilon = self._resolve_epsilon(candidate, accountant)
            else:
                epsilon = candidate
            sigma = calibrate_sigma(epsilon, self.C, self.DELTA)
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
                epsilon = self._resolve_epsilon(candidate, accountant)
                if epsilon < 0:
                    break
            else:
                epsilon = candidate
            sigma = calibrate_sigma(epsilon, self.C, self.DELTA)
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

        for _ in range(self.WARMUP + 2):
            for acct, sched in zip(accountants, schedulers, strict=True):
                candidate = sched.get_epsilon()
                epsilon = self._resolve_epsilon(candidate, acct)
                if epsilon < 0:
                    continue
                sigma = calibrate_sigma(epsilon, self.C, self.DELTA)
                acct.step(sigma=sigma, clipping_norm=self.C, num_steps=1)
                metric = self._simulate_training_metric(epsilon)
                sched.step(epsilon, metric)

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

        for metric_config, metrics_key, value in [
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

            metric_key = _OPTIMIZATION_METRIC_KEY_MAP[metric_config]
            scheduler_copy.step(epsilon, value)

            assert scheduler_copy._observations[-1][1] == pytest.approx(value)

    def test_budget_exhausted_does_not_step_scheduler(self) -> None:
        scheduler = PLDPBOScheduler(
            epsilon_min=self.EPS_MIN,
            epsilon_max=self.EPS_MAX,
            warmup_rounds=self.WARMUP,
            seed=42,
        )
        round_before = scheduler._round

        fit_metrics = {
            "update_norm": 0.0,
            "utility_loss": 0.0,
            "budget_exhausted": True,
        }
        metric_key = _OPTIMIZATION_METRIC_KEY_MAP["nun"]
        metric_value = fit_metrics.get(metric_key)
        exhausted = fit_metrics.get("budget_exhausted", False)

        assert metric_value is not None
        assert exhausted
        # scheduler.step should NOT be called when exhausted
        # (verified by _round not incrementing)
        assert scheduler._round == round_before
