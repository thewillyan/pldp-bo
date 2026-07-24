from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.client import create_client
from src.client.base_client import FlowerClient, _get_optimizer
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
        config.optimizer.momentum = 0.0
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

    def test_momentum_rejected(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient
        config = ExperimentConfig()
        config.privacy.enabled = True
        config.privacy.clipping_mode = "per_example"
        config.optimizer.momentum = 0.9  # should be rejected
        model = _SimpleModel()
        loader = _make_loader()
        with pytest.raises(ValueError, match="momentum"):
            PerExampleDPClient(model, loader, loader, config, client_epsilon=1.0)

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
