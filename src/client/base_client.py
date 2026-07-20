from __future__ import annotations

import copy
from typing import Any

import flwr as fl
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config.loader import ExperimentConfig
from src.device import get_device, to_device
from src.models.base import BaseModel


def _get_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    lr = config.optimizer.lr
    if config.optimizer.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
        )
    if config.optimizer.name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.optimizer.weight_decay)
    raise ValueError(f"Unknown optimizer: {config.optimizer.name}")


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: BaseModel,
        trainloader: DataLoader,
        valloader: DataLoader,
        config: ExperimentConfig,
    ):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.config = config

    def get_parameters(self, config: dict[str, Any]) -> list[Any]:  # noqa: ARG002
        return self.model.get_weights()

    def fit(
        self, parameters: list[Any], config: dict[str, Any],
    ) -> tuple[list[Any], int, dict[str, Any]]:
        self.model.set_weights(parameters)
        net = self.model.get_model().to(get_device())
        net.train()

        proximal_mu = config.get("proximal-mu", 0.0)
        global_params = copy.deepcopy(list(net.parameters())) if proximal_mu > 0 else []

        optimizer = _get_optimizer(net, self.config)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.config.federated.local_epochs):
            for batch in self.trainloader:
                images, labels = to_device(batch)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)

                if proximal_mu > 0:
                    proximal_term = sum(
                        (w - w_global).norm(2)
                        for w, w_global in zip(net.parameters(), global_params, strict=True)
                    )
                    loss = loss + (proximal_mu / 2) * proximal_term

                loss.backward()
                if self.config.optimizer.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.config.optimizer.gradient_clip_norm,
                    )
                optimizer.step()

        return self.model.get_weights(), len(self.trainloader.dataset), {}

    def evaluate(
        self, parameters: list[Any], config: dict[str, Any],  # noqa: ARG002
    ) -> tuple[float, int, dict[str, Any]]:
        self.model.set_weights(parameters)
        net = self.model.get_model().to(get_device())
        net.eval()

        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_images, batch_labels in self.valloader:
                images, labels = to_device((batch_images, batch_labels))
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = loss / len(self.valloader)
        return avg_loss, len(self.valloader.dataset), {"accuracy": accuracy}
