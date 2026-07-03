from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

from src.client.base_client import FlowerClient, _get_optimizer
from src.config.loader import ExperimentConfig
from src.models.base import BaseModel


class DPClient(FlowerClient):
    def __init__(
        self,
        model: BaseModel,
        trainloader: DataLoader,
        valloader: DataLoader,
        config: ExperimentConfig,
    ):
        super().__init__(model, trainloader, valloader, config)
        self._privacy_engine: PrivacyEngine | None = None

    def fit(
        self, parameters: list[Any], config: dict[str, Any]
    ) -> tuple[list[Any], int, dict[str, Any]]:
        self.model.set_weights(parameters)
        net = self.model.get_model()
        net.train()

        optimizer = _get_optimizer(net, self.config)
        criterion = nn.CrossEntropyLoss()

        privacy_engine = PrivacyEngine(secure_mode=False)
        net, optimizer, trainloader = privacy_engine.make_private(
            module=net,
            optimizer=optimizer,
            data_loader=self.trainloader,
            noise_multiplier=self.config.privacy.noise_multiplier,
            max_grad_norm=self.config.privacy.max_grad_norm,
        )
        self._privacy_engine = privacy_engine

        for _ in range(self.config.federated.local_epochs):
            for images, labels in trainloader:
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        epsilon = privacy_engine.get_epsilon(delta=self.config.privacy.delta)
        metrics = {"epsilon": epsilon, "noise_multiplier": self.config.privacy.noise_multiplier}

        return self.model.get_weights(), len(self.trainloader.dataset), metrics

    def get_privacy_spent(self) -> dict[str, float]:
        if self._privacy_engine is None:
            return {"epsilon": 0.0}
        return {"epsilon": self._privacy_engine.get_epsilon(delta=self.config.privacy.delta)}
