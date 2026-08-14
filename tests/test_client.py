from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


def _make_loader() -> DataLoader:
    data = TensorDataset(torch.randn(4, 10), torch.randint(0, 2, (4,)))
    return DataLoader(data, batch_size=2)


def _identity_noise(
    grads: dict[str, torch.Tensor],
    sigma: float,  # noqa: ARG001
    clip_norm: float,  # noqa: ARG001
    rng: np.random.RandomState,  # noqa: ARG001
) -> dict[str, torch.Tensor]:
    return grads


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
        client = create_client(0, model, loader, loader, config,
                               client_epsilon=1.0)
        assert isinstance(client, PerUpdateDPClient)

    def test_creates_per_example_client_when_mode_per_example(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient
        config = ExperimentConfig()
        config.privacy.enabled = True
        config.privacy.clipping_mode = "per_example"
        config.optimizer.momentum = 0.9
        model = _SimpleModel()
        loader = _make_loader()
        client = create_client(0, model, loader, loader, config,
                               client_epsilon=1.0)
        assert isinstance(client, PerExampleDPClient)


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
            model, loader, loader, config,
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
            model, loader, loader, config, client_epsilon=1.0,
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
            model, loader, loader, config,
            client_epsilon=0.5, seed=42,
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
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient
        monkeypatch.setattr(
            "src.client.per_example_dp_client._add_noise", _identity_noise,
        )
        config = self._make_config()
        config.optimizer.momentum = 0.9
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model, loader, loader, config,
            client_epsilon=1.0, seed=42,
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
                    net_ref, images, labels, criterion,
                )
                clipped, _ = _clip_per_example(
                    per_example_grads, config.privacy.update_clip_norm,
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
            model, loader, loader, config,
            client_epsilon=1.0, seed=42,
        )
        params = client.get_parameters({})
        first, _, _ = client.fit(params, {})
        second, _, _ = client.fit(params, {})
        for w1, w2 in zip(first, second, strict=True):
            assert np.allclose(w1, w2, atol=1e-12)

    def test_proximal_shifted_grads_before_clipping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient
        monkeypatch.setattr(
            "src.client.per_example_dp_client._add_noise", _identity_noise,
        )
        config = self._make_config()
        config.optimizer.momentum = 0.0
        config.federated.proximal_mu = 0.01
        config.privacy.update_clip_norm = 0.1  # forces clipping, distinguishes
        config.federated.local_epochs = 1      # shift-before-clip vs after-clip
        model = _SimpleModel()
        loader = _make_loader()
        client = PerExampleDPClient(
            model, loader, loader, config,
            client_epsilon=1.0, seed=42,
        )
        params = client.get_parameters({})

        mu = config.federated.proximal_mu
        criterion = nn.CrossEntropyLoss()
        net_ref = copy.deepcopy(model.get_model())
        global_params = copy.deepcopy(dict(net_ref.named_parameters()))
        opt = torch.optim.SGD(net_ref.parameters(), lr=config.optimizer.lr)
        for images, labels in loader:
            per_example_grads = _compute_per_example_grads(
                net_ref, images, labels, criterion,
            )
            params_now = dict(net_ref.named_parameters())
            shifted = {
                k: g + mu * (params_now[k] - global_params[k])
                for k, g in per_example_grads.items()
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
            model, loader, loader, config,
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
            model, loader, loader, config,
            client_epsilon=0.0,  # exhausted
        )
        params = client.get_parameters({})
        weights, num_examples, metrics = client.fit(params, {})
        assert num_examples == 0
        assert metrics["budget_exhausted"] is True


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
