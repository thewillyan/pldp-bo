from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader

from src.client.base_client import FlowerClient, _get_optimizer
from src.config.loader import ExperimentConfig
from src.device import get_device, to_device
from src.models.base import BaseModel
from src.privacy.accountant import RDPAccountant
from src.privacy.metrics import compute_validation_stats
from src.privacy.per_update_dp import calibrate_sigma_dp_sgd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-example gradient helpers
# ---------------------------------------------------------------------------


def _compute_per_example_grads(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> dict[str, torch.Tensor]:
    """Compute per-example gradients via vmap + grad."""
    params: dict[str, torch.Tensor] = dict(model.named_parameters())
    buffers: dict[str, torch.Tensor] = dict(model.named_buffers())

    def loss_fn(
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        sample: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        out = functional_call(model, (params, buffers), sample.unsqueeze(0))
        return criterion(out, target.unsqueeze(0))  # type: ignore[no-any-return]

    grads_fn = vmap(grad(loss_fn), in_dims=(None, None, 0, 0))
    return grads_fn(params, buffers, inputs, targets)  # type: ignore[no-any-return]


def _clip_per_example(
    grads: dict[str, torch.Tensor],
    clip_norm: float,
) -> tuple[dict[str, torch.Tensor], float]:
    """Clip each example's gradient to *clip_norm*.

    Returns (clipped_grads, clip_fraction) where clip_fraction is the
    fraction of examples whose L2 norm exceeded *clip_norm*.
    """
    # Compute per-example L2 norms across all parameter tensors.
    # grads values have shape (batch, *param_shape).
    flat = torch.cat([g.reshape(g.shape[0], -1) for g in grads.values()], dim=1)
    norms = torch.norm(flat, dim=1)  # (batch,)
    clip_mask = norms > clip_norm
    clip_fraction = float(clip_mask.float().mean())

    # Avoid division by zero for examples with zero norm.
    # clamp min to 1 so unclipped examples stay unchanged.
    scale = torch.clamp(clip_norm / torch.clamp(norms, min=1.0), max=1.0)
    scale = torch.where(clip_mask, scale, torch.ones_like(scale))

    clipped = {k: v * scale.view(-1, *([1] * (v.ndim - 1))) for k, v in grads.items()}
    return clipped, clip_fraction


def _average_grads(
    grads: dict[str, torch.Tensor],
    batch_size: int,  # noqa: ARG001
) -> dict[str, torch.Tensor]:
    """Average clipped gradients over the batch (reduces batch dim)."""
    return {k: v.mean(dim=0) for k, v in grads.items()}


def _add_noise(
    grads: dict[str, torch.Tensor],
    sigma: float,
    clip_norm: float,
    rng: np.random.RandomState,
) -> dict[str, torch.Tensor]:
    """Add Gaussian noise N(0, (sigma * C)^2 I) to averaged gradients."""
    noise_std = sigma * clip_norm
    noisy = {}
    for k, v in grads.items():
        noise = torch.tensor(
            rng.normal(0, noise_std, size=v.shape),
            dtype=v.dtype,
            device=v.device,
        )
        noisy[k] = v + noise
    return noisy


def _set_model_grads(model: nn.Module, grads: dict[str, torch.Tensor]) -> None:
    """Set .grad on each parameter from a grads dict."""
    for name, param in model.named_parameters():
        param.grad = grads[name]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PerExampleDPClient(FlowerClient):
    """Federated client using per-example (DP-SGD) gradient clipping."""

    def __init__(
        self,
        model: BaseModel,
        trainloader: DataLoader,
        valloader: DataLoader,
        config: ExperimentConfig,
        client_epsilon: float | None = None,
        computed_sigma: float | None = None,
        accountant: RDPAccountant | None = None,
        seed: int | None = None,
        remaining_budget: float | None = None,
    ) -> None:
        super().__init__(model, trainloader, valloader, config)

        if config.optimizer.momentum > 0:
            raise ValueError(
                "PerExampleDPClient requires optimizer.momentum == 0 "
                f"(got {config.optimizer.momentum}). "
                "Momentum is not compatible with per-example DP-SGD."
            )

        self._client_epsilon = client_epsilon
        self._computed_sigma = computed_sigma
        self._accountant = accountant
        self._remaining_budget = remaining_budget
        self._seed = seed or config.seed

        total_samples = len(trainloader.dataset)  # type: ignore[arg-type]
        self._sampling_rate = config.data.batch_size / total_samples
        self._total_steps_per_round = (
            config.federated.local_epochs * len(trainloader)
        )

    def _check_budget(self) -> bool:
        if self._client_epsilon is not None and self._client_epsilon == 0:
            logger.warning(
                "Client budget exhausted: epsilon=%.4f",
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
                "utility_efficiency": 0.0,
                "snr": 0.0,
                "sigma": 0.0,
                "utility_loss_clean": 0.0,
                "utility_retention": 0.0,
                "utility_per_remaining": 0.0,
                "agreement": 0.0,
                "budget_exhausted": True,
                "per_example_clip_fraction": 0.0,
                "grad_norm_before_clip": 0.0,
                "grad_norm_after_clip": 0.0,
                "num_opt_steps": 0,
            }
            return parameters, 0, metrics

        self.model.set_weights(parameters)
        net = self.model.get_model().to(get_device())
        criterion = nn.CrossEntropyLoss()
        net.train()

        if self._client_epsilon is not None:
            epsilon = self._client_epsilon
        elif self.config.privacy.target_epsilon is not None:
            epsilon = self.config.privacy.target_epsilon
        else:
            raise ValueError(
                "No epsilon source for PerExampleDPClient. "
                "Provide client_epsilon or set privacy.target_epsilon in config."
            )

        clip_norm = self.config.privacy.update_clip_norm
        delta = self.config.privacy.delta

        if self._computed_sigma is not None and self._computed_sigma > 0:
            sigma = self._computed_sigma
        else:
            sigma = calibrate_sigma_dp_sgd(epsilon, self._sampling_rate, delta)

        optimizer = _get_optimizer(net, self.config)
        rng = np.random.RandomState(self._seed)

        clip_fractions: list[float] = []
        grad_norms_before: list[float] = []
        grad_norms_after: list[float] = []

        for _ in range(self.config.federated.local_epochs):
            for batch in self.trainloader:
                images, labels = to_device(batch)

                per_example_grads = _compute_per_example_grads(
                    net, images, labels, criterion,
                )

                # Stats before clipping
                flat_before = torch.cat(
                    [g.reshape(g.shape[0], -1) for g in per_example_grads.values()],
                    dim=1,
                )
                norms_before = torch.norm(flat_before, dim=1)
                grad_norms_before.append(norms_before.detach().mean().item())

                clipped, clip_frac = _clip_per_example(per_example_grads, clip_norm)
                clip_fractions.append(clip_frac)

                # Stats after clipping
                flat_after = torch.cat(
                    [g.reshape(g.shape[0], -1) for g in clipped.values()],
                    dim=1,
                )
                norms_after = torch.norm(flat_after, dim=1)
                grad_norms_after.append(norms_after.detach().mean().item())

                avg_grad = _average_grads(clipped, images.shape[0])
                noisy_grad = _add_noise(avg_grad, sigma, clip_norm, rng)

                optimizer.zero_grad()
                _set_model_grads(net, noisy_grad)
                optimizer.step()

        local_weights = self.model.get_weights()
        net_eval = self.model.get_model().to(get_device())
        utility_loss_clean, clean_logits = compute_validation_stats(
            net_eval, self.valloader, criterion,
        )

        if self._accountant is not None:
            self._accountant.step(
                sigma=sigma,
                clipping_norm=self._sampling_rate,
                num_steps=self._total_steps_per_round,
                mode="per_example",
            )
            cumulative_epsilon = self._accountant.get_epsilon()
        else:
            cumulative_epsilon = 0.0

        update_norm = 0.0
        noisy_weights = local_weights

        utility_loss_noisy, noisy_logits = compute_validation_stats(
            self.model.get_model(), self.valloader, criterion,
        )
        loss_degradation = max(0.0, utility_loss_noisy - utility_loss_clean)
        inv_loss_clean = 1.0 / max(utility_loss_clean, 1e-12)
        utility_efficiency = -loss_degradation * inv_loss_clean / max(epsilon, 1e-12)
        utility_retention = utility_loss_noisy * inv_loss_clean

        epsilon_remaining = (
            self._remaining_budget
            if self._remaining_budget is not None and self._remaining_budget > 0
            else epsilon
        )
        utility_per_remaining = -loss_degradation * inv_loss_clean / max(epsilon_remaining, 1e-12)

        clean_flat = clean_logits.view(clean_logits.size(0), -1)
        noisy_flat_logits = noisy_logits.view(noisy_logits.size(0), -1)
        cos_sim = torch.nn.functional.cosine_similarity(
            clean_flat, noisy_flat_logits, dim=1,
        )
        agreement = 1.0 - cos_sim.mean().item()

        # Compute SNR from mean gradient norms
        mean_before = float(np.mean(grad_norms_before)) if grad_norms_before else 0.0
        mean_after = float(np.mean(grad_norms_after)) if grad_norms_after else 0.0
        snr = (mean_after ** 2) / max((sigma * clip_norm) ** 2, 1e-12)

        clipped_fraction = float(np.mean(clip_fractions)) if clip_fractions else 0.0

        metrics = {
            "epsilon": epsilon,
            "cumulative_epsilon": cumulative_epsilon,
            "client_epsilon": self._client_epsilon or 0.0,
            "update_norm": update_norm,
            "utility_loss": utility_loss_noisy,
            "utility_efficiency": utility_efficiency,
            "snr": snr,
            "sigma": sigma,
            "utility_loss_clean": utility_loss_clean,
            "utility_retention": utility_retention,
            "utility_per_remaining": utility_per_remaining,
            "agreement": agreement,
            "budget_exhausted": False,
            "per_example_clip_fraction": clipped_fraction,
            "grad_norm_before_clip": mean_before,
            "grad_norm_after_clip": mean_after,
            "num_opt_steps": self._total_steps_per_round,
        }

        return noisy_weights, len(self.trainloader.dataset), metrics  # type: ignore[arg-type]
