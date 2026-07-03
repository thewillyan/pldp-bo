from __future__ import annotations

import torch

from src.config.loader import ModelConfig
from src.models import create_model
from src.models.cnn import CNNModel
from src.models.mlp import MLPModel


def test_create_cnn() -> None:
    config = ModelConfig(name="cnn", num_classes=10)
    model = create_model(config)
    assert isinstance(model, CNNModel)


def test_create_mlp() -> None:
    config = ModelConfig(name="mlp", num_classes=10)
    model = create_model(config)
    assert isinstance(model, MLPModel)


def test_weight_roundtrip() -> None:
    config = ModelConfig(name="cnn", num_classes=10)
    model = create_model(config)

    original_weights = model.get_weights()
    assert len(original_weights) > 0

    modified_weights = [w + 1.0 for w in original_weights]
    model.set_weights(modified_weights)

    new_weights = model.get_weights()
    for orig, new in zip(original_weights, new_weights):
        assert not torch.allclose(
            torch.from_numpy(orig), torch.from_numpy(new)
        )

    model.set_weights(original_weights)
    restored_weights = model.get_weights()
    for orig, restored in zip(original_weights, restored_weights):
        assert torch.allclose(
            torch.from_numpy(orig), torch.from_numpy(restored)
        )


def test_forward_pass() -> None:
    config = ModelConfig(name="cnn", num_classes=10)
    model = create_model(config)
    x = torch.randn(1, 3, 32, 32)
    output = model.get_model()(x)
    assert output.shape == (1, 10)
