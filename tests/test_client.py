from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.client import create_client
from src.client.base_client import _get_optimizer, FlowerClient
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
