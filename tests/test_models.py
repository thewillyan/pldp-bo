from __future__ import annotations

import pytest
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


def test_cnn_femnist_fc_input_3136() -> None:
    model = CNNModel(num_classes=62, dataset_name="femnist")
    assert model._model.fc1.in_features == 3136


def test_cnn_femnist_forward_pass() -> None:
    model = CNNModel(num_classes=62, dataset_name="femnist")
    x = torch.randn(2, 1, 28, 28)
    output = model.get_model()(x)
    assert output.shape == (2, 62)


def test_create_model_cnn_accepts_femnist() -> None:
    config = ModelConfig(name="cnn", num_classes=62)
    model = create_model(config, dataset_name="femnist")
    assert isinstance(model, CNNModel)


def test_create_model_mlp_rejects_femnist() -> None:
    config = ModelConfig(name="mlp", num_classes=62)
    with pytest.raises(ValueError, match="not compatible"):
        create_model(config, dataset_name="femnist")


def test_mlp_femnist_input_size() -> None:
    model = MLPModel(num_classes=62, dataset_name="femnist")
    assert model._model.fc1.in_features == 28 * 28


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
