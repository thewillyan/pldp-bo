from __future__ import annotations

import copy
import logging
import math
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

from src.client.base_client import FlowerClient, _get_optimizer
from src.config.loader import ExperimentConfig
from src.models.base import BaseModel
from src.privacy.accountant import RDPAccountant
from src.privacy.analysis import find_noise_for_target_epsilon
from src.utils import set_seed

logger = logging.getLogger(__name__)


class DPClient(FlowerClient):
    def __init__(
        self,
        model: BaseModel,
        trainloader: DataLoader,
        valloader: DataLoader,
        config: ExperimentConfig,
        client_epsilon: float | None = None,
        accountant: RDPAccountant | None = None,
        num_rounds: int | None = None,
    ):
        super().__init__(model, trainloader, valloader, config)
        self._privacy_engine: PrivacyEngine | None = None
        self._client_epsilon = client_epsilon
        self._accountant = accountant
        self._num_rounds = num_rounds

    def _compute_noise_multiplier(self) -> float:
        if self._client_epsilon is None:
            return self.config.privacy.noise_multiplier

        sampling_rate = self.config.data.batch_size / len(self.trainloader.dataset)
        local_steps = self.config.federated.local_epochs * math.ceil(
            len(self.trainloader.dataset) / self.config.data.batch_size
        )

        return find_noise_for_target_epsilon(
            target_epsilon=self._client_epsilon,
            num_rounds=self._num_rounds or self.config.federated.num_rounds,
            sampling_rate=sampling_rate,
            local_steps=local_steps,
            delta=self.config.privacy.delta,
        )

    def fit(
        self, parameters: list[Any], config: dict[str, Any]
    ) -> tuple[list[Any], int, dict[str, Any]]:
        if self._accountant is not None and self._client_epsilon is not None:
            cumulative_epsilon = self._accountant.get_epsilon()
            if cumulative_epsilon >= self._client_epsilon:
                logger.warning(
                    "Client budget exhausted: cumulative epsilon %.4f >= budget %.4f. "
                    "Returning current model unchanged.",
                    cumulative_epsilon,
                    self._client_epsilon,
                )
                metrics = {
                    "epsilon": 0.0,
                    "cumulative_epsilon": cumulative_epsilon,
                    "client_epsilon": self._client_epsilon,
                    "budget_exhausted": True,
                    "noise_multiplier": 0.0,
                }
                return self.model.get_weights(), 0, metrics

        self.model.set_weights(parameters)
        net = self.model.get_model()
        net.train()

        proximal_mu = config.get("proximal-mu", 0.0)
        global_params = copy.deepcopy(list(net.parameters())) if proximal_mu > 0 else []

        noise_multiplier = self._compute_noise_multiplier()

        optimizer = _get_optimizer(net, self.config)
        criterion = nn.CrossEntropyLoss()

        set_seed(self.config.seed, deterministic=self.config.deterministic)

        privacy_engine = PrivacyEngine(secure_mode=False)
        net, optimizer, trainloader = privacy_engine.make_private(
            module=net,
            optimizer=optimizer,
            data_loader=self.trainloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=self.config.privacy.max_grad_norm,
        )
        self._privacy_engine = privacy_engine

        for _ in range(self.config.federated.local_epochs):
            for images, labels in trainloader:
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)

                if proximal_mu > 0:
                    proximal_term = sum(
                        (w - w_global).norm(2)
                        for w, w_global in zip(net.parameters(), global_params)
                    )
                    loss = loss + (proximal_mu / 2) * proximal_term

                loss.backward()
                optimizer.step()

        epsilon = privacy_engine.get_epsilon(delta=self.config.privacy.delta)

        if self._accountant is not None:
            sampling_rate = self.config.data.batch_size / len(self.trainloader.dataset)
            local_steps = self.config.federated.local_epochs * math.ceil(
                len(self.trainloader.dataset) / self.config.data.batch_size
            )
            self._accountant.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sampling_rate,
                num_steps=local_steps,
            )
            cumulative_epsilon = self._accountant.get_epsilon()
        else:
            cumulative_epsilon = epsilon

        metrics = {
            "epsilon": epsilon,
            "cumulative_epsilon": cumulative_epsilon,
            "client_epsilon": self._client_epsilon or 0.0,
            "noise_multiplier": noise_multiplier,
            "budget_exhausted": False,
        }

        return self.model.get_weights(), len(self.trainloader.dataset), metrics

    def get_privacy_spent(self) -> dict[str, float]:
        if self._accountant is not None:
            return self._accountant.get_privacy_spent()
        if self._privacy_engine is None:
            return {"epsilon": 0.0}
        return {"epsilon": self._privacy_engine.get_epsilon(delta=self.config.privacy.delta)}
