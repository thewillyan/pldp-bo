from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.client.base_client import FlowerClient, _get_optimizer
from src.config.loader import ExperimentConfig
from src.device import get_device, to_device
from src.models.base import BaseModel
from src.privacy.accountant import RDPAccountant
from src.privacy.metrics import compute_utility_loss
from src.privacy.per_update_dp import PerUpdateGaussianMechanism

logger = logging.getLogger(__name__)


class PerUpdateDPClient(FlowerClient):
    def __init__(
        self,
        model: BaseModel,
        trainloader: DataLoader,
        valloader: DataLoader,
        config: ExperimentConfig,
        client_epsilon: float | None = None,
        accountant: RDPAccountant | None = None,
        total_budget: float | None = None,
        seed: int | None = None,
        mechanism_state: dict | None = None,
    ) -> None:
        super().__init__(model, trainloader, valloader, config)
        self._client_epsilon = client_epsilon
        self._accountant = accountant
        self._total_budget = total_budget
        if mechanism_state:
            self._mechanism = PerUpdateGaussianMechanism.from_state(
                mechanism_state,
                clipping_norm=config.privacy.update_clip_norm,
                delta=config.privacy.delta,
            )
        else:
            self._mechanism = PerUpdateGaussianMechanism(
                clipping_norm=config.privacy.update_clip_norm,
                delta=config.privacy.delta,
                seed=seed or config.seed,
            )

    def _check_budget(self) -> bool:
        if self._accountant is not None:
            if self._total_budget is not None:
                cumulative = self._accountant.get_epsilon()
                if cumulative >= self._total_budget:
                    logger.warning(
                        "Client budget exhausted: cumulative ε=%.4f >= total budget ε=%.4f",
                        cumulative,
                        self._total_budget,
                    )
                    return True
            if self._client_epsilon is not None and self._client_epsilon <= 0:
                logger.warning(
                    "Client budget exhausted: epsilon=%.4f (≤ 0)",
                    self._client_epsilon,
                )
                return True
        return False

    def fit(
        self, parameters: list[Any], config: dict[str, Any],
    ) -> tuple[list[Any], int, dict[str, Any]]:
        if self._check_budget():
            metrics = {
                "epsilon": 0.0,
                "cumulative_epsilon": self._accountant.get_epsilon() if self._accountant else 0.0,
                "client_epsilon": self._client_epsilon or 0.0,
                "update_norm": 0.0,
                "utility_loss": 0.0,
                "sigma": 0.0,
                "budget_exhausted": True,
            }
            return parameters, 0, metrics

        self.model.set_weights(parameters)
        global_model_state = copy.deepcopy(self.model.get_model().state_dict())
        net = self.model.get_model().to(get_device())
        net.train()

        proximal_mu = config.get("proximal-mu", self.config.federated.proximal_mu)
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
                optimizer.step()

        if self._client_epsilon is not None:
            epsilon = self._client_epsilon
        elif self.config.privacy.target_epsilon is not None:
            epsilon = self.config.privacy.target_epsilon
        else:
            epsilon = 1.0

        local_weights = self.model.get_weights()
        global_weights = [
            v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            for v in global_model_state.values()
        ]
        delta = [lw - gw for lw, gw in zip(local_weights, global_weights, strict=True)]

        flat_delta = np.concatenate([d.ravel() for d in delta])
        noisy_flat, sigma = self._mechanism.apply(flat_delta, epsilon)

        if self._accountant is not None:
            self._accountant.step(
                sigma=sigma,
                clipping_norm=self._mechanism.clipping_norm,
                num_steps=1,
            )
            cumulative_epsilon = self._accountant.get_epsilon()
        else:
            cumulative_epsilon = 0.0

        update_norm = float(np.linalg.norm(noisy_flat))

        noisy_weights = []
        offset = 0
        for w in delta:
            size = w.size
            noisy_w = noisy_flat[offset : offset + size].reshape(w.shape)
            noisy_weights.append(
                global_weights[len(noisy_weights)] + noisy_w,
            )
            offset += size

        self.model.set_weights(noisy_weights)
        utility_loss = compute_utility_loss(
            self.model.get_model(), self.valloader, criterion,
        )

        metrics = {
            "epsilon": epsilon,
            "cumulative_epsilon": cumulative_epsilon,
            "client_epsilon": self._client_epsilon or 0.0,
            "update_norm": update_norm,
            "utility_loss": utility_loss,
            "sigma": sigma,
            "budget_exhausted": False,
        }

        return noisy_weights, len(self.trainloader.dataset), metrics

    def get_mechanism_state(self) -> dict:
        return self._mechanism.get_state()
