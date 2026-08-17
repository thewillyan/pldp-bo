from __future__ import annotations

import copy
import logging
from typing import Any, Sized, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.client.base_client import FlowerClient, _get_optimizer
from src.config.loader import ExperimentConfig
from src.device import get_device, to_device
from src.models.base import BaseModel
from src.privacy.accountant import RDPAccountant
from src.privacy.metrics import compute_validation_stats
from src.privacy.per_update_dp import PerUpdateGaussianMechanism, compute_rdp_cost

logger = logging.getLogger(__name__)


class PerUpdateDPClient(FlowerClient):
    def __init__(
        self,
        model: BaseModel,
        trainloader: DataLoader[Any],
        valloader: DataLoader[Any],
        config: ExperimentConfig,
        client_epsilon: float | None = None,
        computed_sigma: float | None = None,
        accountant: RDPAccountant | None = None,
        seed: int | None = None,
        mechanism_state: dict[str, Any] | None = None,
        remaining_budget: float | None = None,
        remaining_rdp: float | None = None,
    ) -> None:
        super().__init__(model, trainloader, valloader, config)
        self._client_epsilon = client_epsilon
        self._computed_sigma = computed_sigma
        self._accountant = accountant
        self._remaining_budget = remaining_budget
        self._remaining_rdp = remaining_rdp
        self._rdp_native = config.privacy.accountant_mode == "rdp_native"
        self._rdp_alpha = config.privacy.rdp_alpha
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
        if self._client_epsilon is not None and self._client_epsilon == 0:
            logger.warning(
                "Client budget exhausted: privacy_param=%.6f",
                self._client_epsilon,
            )
            return True
        return False

    def _make_empty_metrics(self, budget_exhausted: bool) -> dict[str, Any]:
        if self._rdp_native:
            return {
                "rdp_cost": 0.0,
                "cumulative_rdp": (
                    self._accountant.get_rdp_at_alpha(self._rdp_alpha) if self._accountant else 0.0
                ),
                "client_rdp": self._client_epsilon or 0.0,
                "update_norm": 0.0,
                "update_norm_clean": 0.0,
                "utility_loss": 0.0,
                "utility_efficiency": 0.0,
                "snr": 0.0,
                "sigma": 0.0,
                "utility_loss_clean": 0.0,
                "utility_retention": 0.0,
                "utility_per_remaining": 0.0,
                "logit_disagreement": 0.0,
                "budget_exhausted": budget_exhausted,
            }
        return {
            "epsilon": 0.0,
            "cumulative_epsilon": self._accountant.get_epsilon() if self._accountant else 0.0,
            "client_epsilon": self._client_epsilon or 0.0,
            "update_norm": 0.0,
            "update_norm_clean": 0.0,
            "utility_loss": 0.0,
            "utility_efficiency": 0.0,
            "snr": 0.0,
            "sigma": 0.0,
            "utility_loss_clean": 0.0,
            "utility_retention": 0.0,
            "utility_per_remaining": 0.0,
            "logit_disagreement": 0.0,
            "budget_exhausted": budget_exhausted,
        }

    def fit(
        self,
        parameters: list[Any],
        config: dict[str, Any],
    ) -> tuple[list[Any], int, dict[str, Any]]:
        del config
        if self._check_budget():
            return parameters, 0, self._make_empty_metrics(budget_exhausted=True)

        if len(self.trainloader) == 0:
            logger.warning("Client has empty trainloader; skipping training")
            return parameters, 0, self._make_empty_metrics(budget_exhausted=False)

        self.model.set_weights(parameters)
        global_weights = self.model.get_weights()
        net = self.model.get_model().to(get_device())
        criterion = nn.CrossEntropyLoss()
        net.train()

        proximal_mu = self.config.federated.proximal_mu
        global_params = copy.deepcopy(list(net.parameters())) if proximal_mu > 0 else []

        optimizer = _get_optimizer(net, self.config)

        for _ in range(self.config.federated.local_epochs):
            for batch in self.trainloader:
                images, labels = to_device(batch)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)

                if proximal_mu > 0:
                    proximal_term = sum(
                        (w - w_global).pow(2).sum()
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

        # privacy_param is either epsilon (epsilon mode) or rdp_cost (rdp_native mode)
        if self._client_epsilon is not None:
            privacy_param = self._client_epsilon
        elif not self._rdp_native and self.config.privacy.target_epsilon is not None:
            privacy_param = self.config.privacy.target_epsilon
        else:
            raise ValueError(
                "No privacy parameter source for PerUpdateDPClient. "
                "Provide client_epsilon or set privacy.target_epsilon in config."
            )

        local_weights = self.model.get_weights()
        net = self.model.get_model().to(get_device())
        utility_loss_clean, clean_logits = compute_validation_stats(
            net,
            self.valloader,
            criterion,
        )

        delta = [lw - gw for lw, gw in zip(local_weights, global_weights, strict=True)]

        flat_delta = np.concatenate([d.ravel() for d in delta])
        noisy_flat, sigma = self._mechanism.apply(
            flat_delta,
            privacy_param,
            sigma=self._computed_sigma,
        )

        delta_norm = float(np.linalg.norm(flat_delta))
        # m_snr = ||Delta||_2^2 / sigma^2 with the clean unclipped update
        # (spec §9.12); delta_norm is the raw update norm, never clipped.
        snr = (delta_norm**2) / max(sigma**2, 1e-12)

        if self._accountant is not None:
            self._accountant.step(
                sigma=sigma,
                clipping_norm=self._mechanism.clipping_norm,
                num_steps=1,
            )
            if self._rdp_native:
                cumulative_privacy = self._accountant.get_rdp_at_alpha(self._rdp_alpha)
            else:
                cumulative_privacy = self._accountant.get_epsilon()
        else:
            cumulative_privacy = 0.0

        update_norm = float(np.linalg.norm(noisy_flat))

        noisy_weights: list[Any] = []
        offset = 0
        for w in delta:
            size = w.size
            noisy_w = noisy_flat[offset : offset + size].reshape(w.shape)
            noisy_weights.append(
                global_weights[len(noisy_weights)] + noisy_w,
            )
            offset += size

        self.model.set_weights(noisy_weights)
        utility_loss_noisy, noisy_logits = compute_validation_stats(
            self.model.get_model(),
            self.valloader,
            criterion,
        )

        loss_degradation = max(0.0, utility_loss_noisy - utility_loss_clean)
        inv_loss_clean = 1.0 / max(utility_loss_clean, 1e-12)
        utility_efficiency = -loss_degradation * inv_loss_clean / max(privacy_param, 1e-12)
        utility_retention = utility_loss_noisy * inv_loss_clean

        privacy_remaining = self._resolve_remaining_rdp()
        utility_per_remaining = -loss_degradation * inv_loss_clean / max(privacy_remaining, 1e-12)

        clean_flat = clean_logits.view(clean_logits.size(0), -1)
        noisy_flat_logits = noisy_logits.view(noisy_logits.size(0), -1)
        cos_sim = torch.nn.functional.cosine_similarity(clean_flat, noisy_flat_logits, dim=1)
        # logit_disagreement = 1 - mean(cos_sim) is the minimization-equivalent
        # complement of the paper's m_agr (maximized logit agreement): minimizing
        # 1 - cos_sim maximizes cos_sim. The report schema (§6.1 meta.display_names)
        # presents this metric as "agreement".
        logit_disagreement = 1.0 - cos_sim.mean().item()

        if self._rdp_native:
            metrics = {
                "rdp_cost": privacy_param,
                "r_t_final": privacy_param,
                "acct_cost": compute_rdp_cost(
                    self._rdp_alpha,
                    sigma,
                    self._mechanism.clipping_norm,
                ),
                "cumulative_rdp": cumulative_privacy,
                "client_rdp": self._client_epsilon or 0.0,
                "update_norm": update_norm,
                "update_norm_clean": delta_norm,
                "utility_loss": utility_loss_noisy,
                "utility_efficiency": utility_efficiency,
                "snr": snr,
                "sigma": sigma,
                "utility_loss_clean": utility_loss_clean,
                "utility_retention": utility_retention,
                "utility_per_remaining": utility_per_remaining,
                "logit_disagreement": logit_disagreement,
                "budget_exhausted": False,
            }
        else:
            metrics = {
                "epsilon": privacy_param,
                "cumulative_epsilon": cumulative_privacy,
                "client_epsilon": self._client_epsilon or 0.0,
                "update_norm": update_norm,
                "update_norm_clean": delta_norm,
                "utility_loss": utility_loss_noisy,
                "utility_efficiency": utility_efficiency,
                "snr": snr,
                "sigma": sigma,
                "utility_loss_clean": utility_loss_clean,
                "utility_retention": utility_retention,
                "utility_per_remaining": utility_per_remaining,
                "logit_disagreement": logit_disagreement,
                "budget_exhausted": False,
            }

        return noisy_weights, len(cast(Sized, self.trainloader.dataset)), metrics

    def get_mechanism_state(self) -> dict[str, Any]:
        return self._mechanism.get_state()
