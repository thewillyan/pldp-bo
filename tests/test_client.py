from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.client import create_client
from src.client.base_client import FlowerClient, _get_optimizer
from src.client.per_example_dp_client import (
    _average_grads,
    _clip_per_example,
    _compute_per_example_grads,
    _set_model_grads,
)
from src.config.loader import ExperimentConfig
from src.models.base import BaseModel


class _SimpleModel(BaseModel):
    def __init__(self) -> None:
        self._net = nn.Linear(10, 2)

    def get_model(self) -> nn.Module:
        return self._net


def _make_loader() -> DataLoader[Any]:
    data = TensorDataset(torch.randn(4, 10), torch.randint(0, 2, (4,)))
    return DataLoader(data, batch_size=2)


def _identity_noise(
    grads: dict[str, torch.Tensor],
    sigma: float,  # noqa: ARG001
    clip_norm: float,  # noqa: ARG001
    rng: np.random.RandomState,  # noqa: ARG001
) -> dict[str, torch.Tensor]:
    return grads


class _CountingOptimizer:
    """Delegate optimizer that counts .step() calls."""

    def __init__(self, optimizer: torch.optim.Optimizer, counter: list[int]) -> None:
        self._optimizer = optimizer
        self._counter = counter

    def step(self) -> None:
        self._counter[0] += 1
        self._optimizer.step()

    def zero_grad(self) -> None:
        self._optimizer.zero_grad()


class TestGetOptimizer:
    def test_sgd(self) -> None:
        config = ExperimentConfig()
        model = nn.Linear(10, 2)
        opt = _get_optimizer(model, config)
        assert isinstance(opt, torch.optim.SGD)

    def test_adam(self) -> None:
        config = ExperimentConfig()
        config.optimizer.name = "adam"
        model = nn.Linear(10, 2)
        opt = _get_optimizer(model, config)
        assert isinstance(opt, torch.optim.Adam)

    def test_unknown_raises(self) -> None:
        config = ExperimentConfig()
        config.optimizer.name = "unknown"
        model = nn.Linear(10, 2)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            _get_optimizer(model, config)

    def test_momentum_override(self) -> None:
        config = ExperimentConfig()
        config.optimizer.momentum = 0.9
        model = nn.Linear(10, 2)
        opt = _get_optimizer(model, config, momentum=0.0)
        assert opt.param_groups[0]["momentum"] == 0.0
        opt_default = _get_optimizer(model, config)
        assert opt_default.param_groups[0]["momentum"] == 0.9


class TestFlowerClient:
    def test_get_parameters_returns_list_of_ndarrays(self) -> None:
        config = ExperimentConfig()
        model = _SimpleModel()
        loader = _make_loader()
        client = FlowerClient(model, loader, loader, config)
        params = client.get_parameters({})
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_evaluate_returns_loss_accuracy(self) -> None:
        config = ExperimentConfig()
        model = _SimpleModel()
        loader = _make_loader()
        client = FlowerClient(model, loader, loader, config)
        params = client.get_parameters({})
        loss, num_examples, metrics = client.evaluate(params, {})
        assert isinstance(loss, float)
        assert isinstance(num_examples, int)
        assert "accuracy" in metrics

    def test_fit_proximal_matches_squared_reference(self) -> None:
        config = ExperimentConfig()
        config.optimizer.momentum = 0.0
        mu = 0.01
        model = _SimpleModel()
        loader = _make_loader()
        client = FlowerClient(model, loader, loader, config)
        params = client.get_parameters({})

        criterion = nn.CrossEntropyLoss()
        net_ref = copy.deepcopy(model.get_model())
        global_params = copy.deepcopy(list(net_ref.parameters()))
        opt = torch.optim.SGD(net_ref.parameters(), lr=config.optimizer.lr)
        for _ in range(config.federated.local_epochs):
            for images, labels in loader:
                opt.zero_grad()
                loss = criterion(net_ref(images), labels)
                loss = loss + (mu / 2) * sum(
                    (w - w_global).pow(2).sum()
                    for w, w_global in zip(net_ref.parameters(), global_params, strict=True)
                )
                loss.backward()
                opt.step()

        _, num_examples, _ = client.fit(params, {"proximal-mu": mu})
        assert num_examples > 0
        trained = model.get_model()
        for p_ref, p_client in zip(net_ref.parameters(), trained.parameters(), strict=True):
            assert torch.allclose(p_ref.detach(), p_client.detach(), atol=1e-6)


class TestCreateClient:
    def test_creates_plain_client_when_privacy_disabled(self) -> None:
        config = ExperimentConfig()
        config.privacy.enabled = False
        model = _SimpleModel()
        loader = _make_loader()
        client = create_client(0, model, loader, loader, config)
        assert isinstance(client, FlowerClient)

    def test_creates_per_update_client_when_mode_per_update(self) -> None:
        from src.client.per_update_dp_client import PerUpdateDPClient

        config = ExperimentConfig()
        config.privacy.enabled = True
        config.privacy.clipping_mode = "per_update"
        model = _SimpleModel()
        loader = _make_loader()
        client = create_client(0, model, loader, loader, config, client_epsilon=1.0)
        assert isinstance(client, PerUpdateDPClient)

    def test_creates_per_example_client_when_mode_per_example(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = ExperimentConfig()
        config.privacy.enabled = True
        config.privacy.clipping_mode = "per_example"
        config.optimizer.momentum = 0.9
        model = _SimpleModel()
        loader = _make_loader()
        client = create_client(0, model, loader, loader, config, client_epsilon=1.0)
        assert isinstance(client, PerExampleDPClient)

    def test_forwards_remaining_rdp_to_dp_clients(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient
        from src.client.per_update_dp_client import PerUpdateDPClient

        model = _SimpleModel()
        loader = _make_loader()
        per_example_cfg = ExperimentConfig()
        per_example_cfg.privacy.enabled = True
        per_example_cfg.privacy.clipping_mode = "per_example"
        per_update_cfg = ExperimentConfig()
        per_update_cfg.privacy.enabled = True
        per_update_cfg.privacy.clipping_mode = "per_update"
        pe = create_client(
            0, model, loader, loader, per_example_cfg, client_epsilon=1.0, remaining_rdp=6.5
        )
        pu = create_client(
            0, model, loader, loader, per_update_cfg, client_epsilon=1.0, remaining_rdp=6.5
        )
        assert isinstance(pe, PerExampleDPClient)
        assert isinstance(pu, PerUpdateDPClient)
        assert pe._remaining_rdp == 6.5
        assert pu._remaining_rdp == 6.5


class TestPerExampleDPClient:
    def _make_config(self) -> ExperimentConfig:
        config = ExperimentConfig()
        config.privacy.enabled = True
        config.privacy.clipping_mode = "per_example"
        config.optimizer.momentum = 0.0
        config.data.batch_size = 2  # match _make_loader
        return config

    def test_fit_returns_valid_output(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = self._make_config()
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
        )
        params = client.get_parameters({})
        weights, num_examples, metrics = client.fit(params, {})
        assert isinstance(weights, list)
        assert all(isinstance(p, np.ndarray) for p in weights)
        assert num_examples > 0
        assert "epsilon" in metrics
        assert "sigma" in metrics
        assert "per_example_clip_fraction" in metrics
        assert "grad_norm_before_clip" in metrics
        assert "grad_norm_after_clip" in metrics
        assert "num_opt_steps" in metrics

    def test_momentum_and_proximal_accepted(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = self._make_config()
        config.optimizer.momentum = 0.9
        config.federated.proximal_mu = 0.01
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
        )
        assert client is not None

    def test_fit_momentum_proximal_rdp_native_runs(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = self._make_config()
        config.optimizer.momentum = 0.9
        config.federated.proximal_mu = 0.01
        config.privacy.accountant_mode = "rdp_native"
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=0.5,
            seed=42,
        )
        params = client.get_parameters({})
        weights, num_examples, metrics = client.fit(params, {})
        assert num_examples > 0
        assert weights
        assert metrics["num_opt_steps"] > 0
        assert "rdp_cost" in metrics
        assert "cumulative_rdp" in metrics
        assert metrics["sigma"] > 0

    def test_momentum_matches_reference_implementation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        monkeypatch.setattr(
            "src.client.per_example_dp_client._add_noise",
            _identity_noise,
        )
        config = self._make_config()
        config.optimizer.momentum = 0.9
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        params = client.get_parameters({})

        criterion = nn.CrossEntropyLoss()
        net_ref = copy.deepcopy(model.get_model())
        opt = torch.optim.SGD(
            net_ref.parameters(),
            lr=config.optimizer.lr,
            momentum=0.9,
        )
        for _ in range(config.federated.local_epochs):
            for images, labels in loader:
                per_example_grads = _compute_per_example_grads(
                    net_ref,
                    images,
                    labels,
                    criterion,
                )
                clipped, _ = _clip_per_example(
                    per_example_grads,
                    config.privacy.update_clip_norm,
                )
                opt.zero_grad()
                _set_model_grads(net_ref, _average_grads(clipped))
                opt.step()

        client.fit(params, {})
        trained = model.get_model()
        for p_ref, p_client in zip(net_ref.parameters(), trained.parameters(), strict=True):
            assert torch.allclose(p_ref.detach(), p_client.detach(), atol=1e-6)

    def test_momentum_buffer_resets_between_fits(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = self._make_config()
        config.optimizer.momentum = 0.9
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        params = client.get_parameters({})
        first, _, _ = client.fit(params, {})
        second, _, _ = client.fit(params, {})
        for w1, w2 in zip(first, second, strict=True):
            assert np.allclose(w1, w2, atol=1e-12)

    def test_proximal_shifted_grads_before_clipping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        monkeypatch.setattr(
            "src.client.per_example_dp_client._add_noise",
            _identity_noise,
        )
        config = self._make_config()
        config.optimizer.momentum = 0.0
        config.federated.proximal_mu = 0.01
        config.privacy.update_clip_norm = 0.1  # forces clipping, distinguishes
        config.federated.local_epochs = 1  # shift-before-clip vs after-clip
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        params = client.get_parameters({})

        mu = config.federated.proximal_mu
        criterion = nn.CrossEntropyLoss()
        net_ref = copy.deepcopy(model.get_model())
        global_params = copy.deepcopy(dict(net_ref.named_parameters()))
        opt = torch.optim.SGD(net_ref.parameters(), lr=config.optimizer.lr)
        for images, labels in loader:
            per_example_grads = _compute_per_example_grads(
                net_ref,
                images,
                labels,
                criterion,
            )
            params_now = dict(net_ref.named_parameters())
            shifted = {
                k: g + mu * (params_now[k] - global_params[k]) for k, g in per_example_grads.items()
            }
            clipped, _ = _clip_per_example(shifted, config.privacy.update_clip_norm)
            opt.zero_grad()
            _set_model_grads(net_ref, _average_grads(clipped))
            opt.step()

        client.fit(params, {})
        trained = model.get_model()
        for p_ref, p_client in zip(net_ref.parameters(), trained.parameters(), strict=True):
            assert torch.allclose(p_ref.detach(), p_client.detach(), atol=1e-6)

    def test_clip_fraction_between_0_and_1(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = self._make_config()
        config.privacy.update_clip_norm = 0.01  # very small clip norm → high clip fraction
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
        )
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        assert 0.0 <= metrics["per_example_clip_fraction"] <= 1.0

    def test_budget_exhausted_returns_zeros(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = self._make_config()
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=0.0,  # exhausted
        )
        params = client.get_parameters({})
        weights, num_examples, metrics = client.fit(params, {})
        assert num_examples == 0
        assert metrics["budget_exhausted"] is True

    def test_clean_pass_runs_for_reference_variant(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = self._make_config()
        config.method = "pldpbo_retention"
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        assert metrics["utility_loss_clean"] > 0
        assert metrics["update_norm_clean"] > 0
        assert metrics["utility_retention"] > 0

    def test_no_clean_pass_for_nun(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = self._make_config()
        config.method = "pldpbo_nun"
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        assert metrics["utility_loss_clean"] == 0.0
        assert metrics["update_norm_clean"] == 0.0
        assert metrics["utility_retention"] == 0.0
        assert metrics["utility_efficiency"] == 0.0
        assert metrics["utility_per_remaining"] == 0.0
        assert metrics["logit_disagreement"] == 0.0

    def test_clean_loss_differs_from_global_model_eval(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient
        from src.privacy.metrics import compute_validation_stats

        config = self._make_config()
        config.method = "pldpbo_retention"
        config.optimizer.lr = 0.5
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        params = client.get_parameters({})
        model.set_weights(params)
        global_loss, _ = compute_validation_stats(
            model.get_model(),
            loader,
            nn.CrossEntropyLoss(),
        )
        _, _, metrics = client.fit(params, {})
        assert not np.isclose(metrics["utility_loss_clean"], global_loss)

    def test_clean_pass_accounting_unaffected(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        reference_config = self._make_config()
        reference_config.method = "pldpbo_retention"
        nun_config = self._make_config()
        nun_config.method = "pldpbo_nun"
        model = _SimpleModel()
        loader = _make_loader()

        reference = PerExampleDPClient(
            _SimpleModel(),
            loader,
            loader,
            reference_config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        nun = PerExampleDPClient(
            model,
            loader,
            loader,
            nun_config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        params = nun.get_parameters({})
        _, _, ref_metrics = reference.fit(params, {})
        _, _, nun_metrics = nun.fit(params, {})
        assert ref_metrics["sigma"] == nun_metrics["sigma"]
        assert ref_metrics["cumulative_epsilon"] == nun_metrics["cumulative_epsilon"]

    def test_clean_loss_varies_per_client(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = self._make_config()
        config.method = "pldpbo_retention"
        config.data.batch_size = 4
        x1 = torch.randn(16, 10)
        y1 = torch.randint(0, 2, (16,))
        x2 = torch.randn(16, 10) + 5.0  # shifted distribution
        y2 = torch.randint(0, 2, (16,))
        loader_a = DataLoader(TensorDataset(x1, y1), batch_size=4)
        loader_b = DataLoader(TensorDataset(x2, y2), batch_size=4)
        params = _SimpleModel().get_weights()

        client_a = PerExampleDPClient(
            _SimpleModel(),
            loader_a,
            loader_a,
            config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        client_b = PerExampleDPClient(
            _SimpleModel(),
            loader_b,
            loader_b,
            config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        _, _, metrics_a = client_a.fit(params, {})
        _, _, metrics_b = client_b.fit(params, {})
        assert not np.isclose(
            metrics_a["utility_loss_clean"],
            metrics_b["utility_loss_clean"],
        )

    def test_clean_pass_doubles_optimizer_steps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.client import per_example_dp_client as mod
        from src.client.per_example_dp_client import PerExampleDPClient

        dp_steps: list[int] = [0]
        clean_steps: list[int] = [0]
        original = mod._get_optimizer  # type: ignore[attr-defined]

        def counting(
            net: nn.Module,
            config: ExperimentConfig,
            momentum: float | None = None,
        ) -> _CountingOptimizer:
            opt = original(net, config, momentum=momentum)
            counter = clean_steps if momentum is None else dp_steps
            return _CountingOptimizer(opt, counter)

        monkeypatch.setattr(mod, "_get_optimizer", counting)
        config = self._make_config()
        config.optimizer.momentum = 0.9
        config.method = "pldpbo_retention"
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        params = client.get_parameters({})
        client.fit(params, {})
        assert dp_steps[0] > 0
        assert clean_steps[0] == dp_steps[0]

    def test_snr_uses_clean_unclipped_update_norm(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = self._make_config()
        config.method = "pldpbo_snr"
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        expected = metrics["update_norm_clean"] ** 2 / max(metrics["sigma"] ** 2, 1e-12)
        assert metrics["snr"] == pytest.approx(expected, rel=1e-9)
        assert metrics["update_norm_clean"] > 0

    def test_snr_zero_without_clean_pass(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = self._make_config()
        config.method = "pldpbo_nun"
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
        )
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        assert metrics["snr"] == 0.0

    def test_missing_remaining_rdp_raises_for_clean_pass(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = self._make_config()
        config.method = "pldpbo_retention"
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
        )
        params = client.get_parameters({})
        with pytest.raises(ValueError, match="remaining_rdp"):
            client.fit(params, {})

    def test_missing_remaining_rdp_ok_for_non_clean_pass(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient

        config = self._make_config()
        config.method = "pldpbo_nun"
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
        )
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        assert metrics["utility_per_remaining"] == 0.0


class TestPerUpdateDPClient:
    def _make_config(self) -> ExperimentConfig:
        config = ExperimentConfig()
        config.privacy.enabled = True
        config.privacy.clipping_mode = "per_update"
        config.data.batch_size = 2  # match _make_loader
        return config

    def test_fit_reports_clean_update_norm(self) -> None:
        from src.client.per_update_dp_client import PerUpdateDPClient

        config = self._make_config()
        model = _SimpleModel()
        loader = _make_loader()
        client = PerUpdateDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        assert metrics["update_norm_clean"] > 0
        assert "update_norm_clean" in metrics

    def test_snr_uses_raw_unclipped_update_norm(self) -> None:
        from src.client.per_update_dp_client import PerUpdateDPClient

        config = self._make_config()
        config.privacy.update_clip_norm = 1e-6  # forces clipping; snr must stay raw
        model = _SimpleModel()
        loader = _make_loader()
        client = PerUpdateDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
            remaining_rdp=5.0,
        )
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        expected = metrics["update_norm_clean"] ** 2 / max(metrics["sigma"] ** 2, 1e-12)
        assert metrics["snr"] == pytest.approx(expected, rel=1e-9)
        assert metrics["update_norm_clean"] > config.privacy.update_clip_norm

    def test_missing_remaining_rdp_raises(self) -> None:
        from src.client.per_update_dp_client import PerUpdateDPClient

        config = self._make_config()
        model = _SimpleModel()
        loader = _make_loader()
        client = PerUpdateDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
        )
        params = client.get_parameters({})
        with pytest.raises(ValueError, match="remaining_rdp"):
            client.fit(params, {})

    def test_rdp_native_reports_r_t_final_and_acct_cost(self) -> None:
        from src.client.per_update_dp_client import PerUpdateDPClient
        from src.privacy.accountant import RDPAccountant
        from src.privacy.per_update_dp import compute_rdp_cost

        config = self._make_config()
        config.privacy.accountant_mode = "rdp_native"
        config.privacy.rdp_alpha = 10.0
        config.privacy.update_clip_norm = 1.0
        model = _SimpleModel()
        loader = _make_loader()
        client = PerUpdateDPClient(
            model,
            loader,
            loader,
            config,
            client_epsilon=1.0,
            seed=42,
            accountant=RDPAccountant(delta=1e-5),
            remaining_rdp=5.0,
        )
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        assert metrics["r_t_final"] == pytest.approx(1.0)
        assert metrics["acct_cost"] == pytest.approx(
            compute_rdp_cost(10.0, metrics["sigma"], 1.0),
            rel=1e-9,
        )


class TestClipPerExample:
    def test_no_op_when_all_norms_below_threshold(self) -> None:
        grads = {"w": torch.tensor([[0.1, 0.2], [0.3, 0.0]])}
        clipped, clip_frac = _clip_per_example(grads, clip_norm=5.0)
        assert torch.allclose(clipped["w"], grads["w"])
        assert clip_frac == 0.0

    def test_clips_to_exact_norm(self) -> None:
        grads = {"w": torch.tensor([[3.0, 4.0], [0.0, 0.0]])}
        clipped, clip_frac = _clip_per_example(grads, clip_norm=1.0)
        norms = torch.linalg.vector_norm(clipped["w"], dim=1)
        # First example: norm was 5.0, clipped to 1.0
        assert abs(norms[0].item() - 1.0) < 1e-5
        # Second example: zero norm, unchanged
        assert torch.allclose(clipped["w"][1], grads["w"][1])
        assert abs(clip_frac - 0.5) < 1e-5

    def test_zero_norm_unchanged(self) -> None:
        grads = {"w": torch.tensor([[0.0, 0.0], [0.0, 0.0]])}
        clipped, clip_frac = _clip_per_example(grads, clip_norm=1.0)
        assert torch.allclose(clipped["w"], grads["w"])
        assert clip_frac == 0.0

    def test_clip_fraction_matches_count(self) -> None:
        grads = {"w": torch.tensor([[1.0, 0.0], [0.0, 0.1], [5.0, 0.0]])}
        clipped, clip_frac = _clip_per_example(grads, clip_norm=1.0)
        # norms: [1.0, 0.1, 5.0] → only last exceeds 1.0
        assert abs(clip_frac - 1.0 / 3.0) < 1e-5

    def test_small_clip_norm_correct_scale(self) -> None:
        grads = {"w": torch.tensor([[0.1, 0.0], [0.5, 0.0], [0.7, 0.0]])}
        clipped, _ = _clip_per_example(grads, clip_norm=0.3)
        norms = torch.linalg.vector_norm(clipped["w"], dim=1)
        # norm 0.1 → below threshold, unchanged, norm stays 0.1
        assert abs(norms[0].item() - 0.1) < 1e-5
        # norm 0.5 → clipped to 0.3
        assert abs(norms[1].item() - 0.3) < 1e-5
        # norm 0.7 → clipped to 0.3
        assert abs(norms[2].item() - 0.3) < 1e-5


class _FixedPredictionModel:
    """Model stub whose Linear(6, 62) predicts class p_i for one-hot input e_i."""

    def __init__(self, predictions: Sequence[int]) -> None:
        net = nn.Linear(6, 62, bias=False)
        with torch.no_grad():
            net.weight.zero_()
            for i, p in enumerate(predictions):
                net.weight[p, i] = 1.0
        self._net = net

    def get_model(self) -> nn.Module:
        return self._net

    def set_weights(self, parameters: list[Any]) -> None:
        pass


class _FakeQueryContext:
    run_config = {"config-path": "unused.yaml"}
    node_config = {"partition-id": "0", "num-partitions": "2"}


class TestClientTestAccuracyQuery:
    def _make_train_dataset(self) -> TensorDataset:
        dataset = TensorDataset(torch.randn(5, 6), torch.tensor([0, 1, 2, 0, 1]))
        cast(Any, dataset).users = torch.tensor([0, 0, 1, 1, 2])
        return dataset

    def _make_test_dataset(self, users: list[int]) -> TensorDataset:
        dataset = TensorDataset(torch.eye(6), torch.tensor([0, 1, 2, 0, 1, 2]))
        cast(Any, dataset).users = torch.tensor(users)
        return dataset

    def _monkeypatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        train_subset: Subset[Any],
        config: ExperimentConfig,
        test_users: list[int] | None = None,
    ) -> None:
        monkeypatch.setattr(
            "src.client_app.load_config",
            lambda path, overrides: config,  # noqa: ARG005
        )
        monkeypatch.setattr(
            "src.client_app.create_client_dataloader",
            lambda _data, _pid, _num, _seed: (None, None, train_subset, None, 5),
        )
        monkeypatch.setattr(
            "src.client_app.create_dataset",
            lambda _data: self._make_train_dataset(),
        )
        users = test_users if test_users is not None else [0, 1, 2, 0, 1, 2]
        monkeypatch.setattr(
            "src.client_app.create_test_dataset",
            lambda _data: self._make_test_dataset(users),
        )
        monkeypatch.setattr(
            "src.client_app.create_dataloaders",
            lambda dataset, batch_size, shuffle: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            ),
        )
        monkeypatch.setattr(
            "src.client_app.create_model",
            lambda _model_cfg, dataset_name: _FixedPredictionModel([0, 1, 9, 9, 1, 9]),  # noqa: ARG005
        )

    def test_evaluates_only_own_writers_test_samples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.client_app import query

        config = ExperimentConfig()
        config.data.name = "femnist"
        train_subset = Subset(self._make_train_dataset(), [0, 1, 2, 3])  # writers {0, 1}
        self._monkeypatch(monkeypatch, train_subset, config)
        msg = Message(
            content=RecordDict(
                {
                    "config": ConfigRecord({"task": "client_test_accuracy"}),
                    "arrays": ArrayRecord(
                        _FixedPredictionModel([0, 1, 9, 9, 1, 9]).get_model().state_dict(),
                    ),
                }
            ),
            message_type="query",
            dst_node_id=0,
            group_id="t",
        )
        reply = query(msg, cast(Any, _FakeQueryContext()))
        meta = reply.content.config_records.get("config", ConfigRecord())
        assert meta.get("partition_id") == 0
        # test samples of writers {0,1}: indices 0,1,3,4; predictions 0,1,9,1 vs labels 0,1,0,1
        assert meta.get("test_accuracy") == pytest.approx(0.75)
        assert meta.get("n_test") == 4

    def test_no_test_samples_for_writer_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.client_app import query

        config = ExperimentConfig()
        config.data.name = "femnist"
        train_subset = Subset(self._make_train_dataset(), [4])  # writer {2} only
        self._monkeypatch(monkeypatch, train_subset, config, test_users=[0, 1, 0, 1, 0, 1])
        msg = Message(
            content=RecordDict(
                {
                    "config": ConfigRecord({"task": "client_test_accuracy"}),
                    "arrays": ArrayRecord({}),
                }
            ),
            message_type="query",
            dst_node_id=0,
            group_id="t",
        )
        reply = query(msg, cast(Any, _FakeQueryContext()))
        meta = reply.content.config_records.get("config", ConfigRecord())
        assert meta.get("test_accuracy") == pytest.approx(0.0)
        assert meta.get("n_test") == 0

    def test_requires_femnist_dataset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.client_app import query

        config = ExperimentConfig()
        config.data.name = "mnist"
        train_subset = Subset(self._make_train_dataset(), [0])
        self._monkeypatch(monkeypatch, train_subset, config)
        msg = Message(
            content=RecordDict(
                {
                    "config": ConfigRecord({"task": "client_test_accuracy"}),
                    "arrays": ArrayRecord({}),
                }
            ),
            message_type="query",
            dst_node_id=0,
            group_id="t",
        )
        with pytest.raises(ValueError, match="femnist"):
            query(msg, cast(Any, _FakeQueryContext()))


class TestReadRemainingRdp:
    def test_reads_float_from_config(self) -> None:
        from src.client_app import _read_remaining_rdp

        msg = Message(
            content=RecordDict({"config": ConfigRecord({"remaining_rdp": 6.5})}),
            message_type="train",
            dst_node_id=0,
        )
        assert _read_remaining_rdp(msg) == 6.5

    def test_none_when_missing(self) -> None:
        from src.client_app import _read_remaining_rdp

        msg = Message(
            content=RecordDict({"config": ConfigRecord({})}),
            message_type="train",
            dst_node_id=0,
        )
        assert _read_remaining_rdp(msg) is None

    def test_none_when_no_config_record(self) -> None:
        from src.client_app import _read_remaining_rdp

        msg = Message(
            content=RecordDict({}),
            message_type="train",
            dst_node_id=0,
        )
        assert _read_remaining_rdp(msg) is None


class _FakeFitClient:
    """Stub client returning a fixed fit metrics dict."""

    def __init__(self, metrics: dict[str, Any]) -> None:
        self._metrics = metrics

    def fit(
        self,
        _parameters: list[Any],
        config: dict[str, Any],  # noqa: ARG002
    ) -> tuple[list[Any], int, dict[str, Any]]:
        return [], 2, self._metrics


class _FakeTrainContext:
    run_config = {"config-path": "unused.yaml"}
    node_config = {"partition-id": "0", "num-partitions": "2"}

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}


class TestTrainReplySpecFields:
    """train() reply carries r_t_candidate / phase / observed_m / bo_time / acct_time."""

    def _make_config(self) -> ExperimentConfig:
        config = ExperimentConfig()
        config.method = "pldpbo_snr"
        config.privacy.enabled = True
        config.privacy.accountant_mode = "rdp_native"
        config.privacy.clipping_mode = "per_example"
        config.privacy.rdp_alpha = 10.0
        config.privacy.total_budget = 10.0
        config.bo.enabled = True
        config.bo.optimization_metric = "nun"
        config.bo.min_warmup = 10
        return config

    def _monkeypatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: ExperimentConfig,
        metrics: dict[str, Any],
    ) -> None:
        monkeypatch.setattr(
            "src.client_app.load_config",
            lambda path, overrides: config,  # noqa: ARG005
        )
        monkeypatch.setattr(
            "src.client_app.assert_locked_config",
            lambda cfg: None,  # noqa: ARG005
        )
        monkeypatch.setattr(
            "src.client_app.create_client_dataloader",
            lambda _data, _pid, _num, _seed: (None, None, list(range(1000)), None, 1000),
        )
        monkeypatch.setattr(
            "src.client_app.create_model",
            lambda _model_cfg, dataset_name: _SimpleModel(),  # noqa: ARG005
        )
        monkeypatch.setattr(
            "src.client_app.create_client",
            lambda **kwargs: _FakeFitClient(metrics),  # noqa: ARG005
        )

    def _reply_metrics(
        self, monkeypatch: pytest.MonkeyPatch, fit_metrics: dict[str, Any]
    ) -> MetricRecord | ConfigRecord:
        from src.client_app import train

        config = self._make_config()
        self._monkeypatch(monkeypatch, config, fit_metrics)
        msg = Message(
            content=RecordDict(
                {
                    "config": ConfigRecord({"per_client_budget": 10.0, "remaining_rdp": 5.0}),
                    "arrays": ArrayRecord(_SimpleModel().get_model().state_dict()),
                }
            ),
            message_type="train",
            dst_node_id=0,
        )
        reply = train(msg, cast(Any, _FakeTrainContext()))
        return reply.content.metric_records.get("metrics", ConfigRecord())

    def test_reports_spec_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fit_metrics = {
            "rdp_cost": 0.05,
            "update_norm": 0.42,
            "cumulative_rdp": 0.05,
            "budget_exhausted": False,
        }
        metrics = self._reply_metrics(monkeypatch, fit_metrics)
        # Warm-up grid round 0: candidate == 0.01 (pre-enforcement),
        # phase code 0.0 = "warmup".
        assert metrics["r_t_candidate"] == pytest.approx(0.01)
        assert metrics["phase"] == pytest.approx(0.0)
        assert metrics["observed_m"] == pytest.approx(0.42)
        bo_time = metrics["bo_time"]
        acct_time = metrics["acct_time"]
        assert isinstance(bo_time, float) and bo_time >= 0
        assert isinstance(acct_time, float) and acct_time >= 0
        assert metrics["num-examples"] == 2
        assert metrics["client-id"] == 0

    def test_exhausted_reports_phase_and_no_observed_m(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fit_metrics = {
            "rdp_cost": 0.0,
            "cumulative_rdp": 10.0,
            "budget_exhausted": True,
        }
        metrics = self._reply_metrics(monkeypatch, fit_metrics)
        # Phase code 2.0 = "exhausted".
        assert metrics["phase"] == pytest.approx(2.0)
        assert "observed_m" not in metrics
