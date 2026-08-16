from __future__ import annotations

import copy
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
from src.privacy.per_update_dp import (
    calibrate_sigma_dp_sgd,
    compute_rdp_cost_dp_sgd,
)

logger = logging.getLogger(__name__)

# Reference variants compute their clean statistics from a locally-trained
# no-DP model (spec §9.6). NUN/Utility (and the fixed baselines) use only the
# privatized model, so they skip the clean pass (≈2x local cost saving).
CLEAN_PASS_METHODS = frozenset(
    {
        "pldpbo_retention",
        "pldpbo_efficiency",
        "pldpbo_perremaining",
        "pldpbo_snr",
        "pldpbo_agreement",
    }
)


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
    flat = torch.cat([g.reshape(g.shape[0], -1) for g in grads.values()], dim=1)
    norms = torch.linalg.vector_norm(flat, dim=1)
    clip_mask = norms > clip_norm
    clip_fraction = float(clip_mask.float().mean())

    scale = torch.clamp(clip_norm / norms.clamp(min=1e-6), max=1.0)
    scale = torch.where(clip_mask, scale, torch.ones_like(scale))

    clipped = {k: v * scale.view(-1, *([1] * (v.ndim - 1))) for k, v in grads.items()}
    return clipped, clip_fraction


def _average_grads(grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


def _run_clean_pass(
    clean_net: nn.Module,
    trainloader: DataLoader[Any],
    config: ExperimentConfig,
    criterion: nn.Module,
) -> tuple[nn.Module, float]:
    """Train a fresh local model from the global weights with no DP noise.

    *clean_net* must be a pristine copy of the global model (the DP pass
    mutates the client's own model in place). Same E, B, lr, momentum and seed
    as the DP pass; no clipping, no noise and no accountant steps — the clean
    pass consumes no privacy budget. Returns (clean model, clean update norm
    ||w_clean - w_global||), consumed by the reference variants and by IMPL-10's
    SNR formula.
    """
    clean_net.train()

    initial_flat = np.concatenate(
        [p.detach().cpu().numpy().ravel() for p in clean_net.parameters()],
    )

    optimizer = _get_optimizer(clean_net, config)
    for _ in range(config.federated.local_epochs):
        for batch in trainloader:
            images, labels = to_device(batch)
            optimizer.zero_grad()
            outputs = clean_net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    final_flat = np.concatenate(
        [p.detach().cpu().numpy().ravel() for p in clean_net.parameters()],
    )
    update_norm_clean = float(np.linalg.norm(final_flat - initial_flat))
    return clean_net, update_norm_clean


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PerExampleDPClient(FlowerClient):
    """Federated client using per-example (DP-SGD) gradient clipping."""

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
        remaining_budget: float | None = None,
        remaining_rdp: float | None = None,
    ) -> None:
        super().__init__(model, trainloader, valloader, config)

        self._client_epsilon = client_epsilon
        self._computed_sigma = computed_sigma
        self._accountant = accountant
        self._remaining_budget = remaining_budget
        self._remaining_rdp = remaining_rdp
        self._seed = seed or config.seed
        self._rdp_native = config.privacy.accountant_mode == "rdp_native"
        self._rdp_alpha = config.privacy.rdp_alpha

        total_samples = len(trainloader.dataset)  # type: ignore[arg-type]
        self._sampling_rate = config.data.batch_size / total_samples
        self._total_steps_per_round = config.federated.local_epochs * len(trainloader)

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
                "per_example_clip_fraction": 0.0,
                "grad_norm_before_clip": 0.0,
                "grad_norm_after_clip": 0.0,
                "num_opt_steps": 0,
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
            "per_example_clip_fraction": 0.0,
            "grad_norm_before_clip": 0.0,
            "grad_norm_after_clip": 0.0,
            "num_opt_steps": 0,
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
        net = self.model.get_model().to(get_device())
        criterion = nn.CrossEntropyLoss()
        net.train()

        proximal_mu = self.config.federated.proximal_mu
        global_params = copy.deepcopy(dict(net.named_parameters())) if proximal_mu > 0 else {}
        clean_pass = self.config.method in CLEAN_PASS_METHODS
        # Pristine global copy for the clean pass: the DP pass mutates *net*
        # in place, so the copy must be taken before training starts.
        clean_net_ref = copy.deepcopy(net) if clean_pass else None

        initial_flat = np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in net.parameters()],
        )

        # privacy_param is either epsilon (epsilon mode) or rdp_cost (rdp_native mode)
        if self._client_epsilon is not None:
            privacy_param = self._client_epsilon
        elif not self._rdp_native and self.config.privacy.target_epsilon is not None:
            privacy_param = self.config.privacy.target_epsilon
        else:
            raise ValueError(
                "No privacy parameter source for PerExampleDPClient. "
                "Provide client_epsilon or set privacy.target_epsilon in config."
            )

        clip_norm = self.config.privacy.update_clip_norm
        delta = self.config.privacy.delta

        if self._computed_sigma is not None and self._computed_sigma > 0:
            sigma = self._computed_sigma
        else:
            sigma = calibrate_sigma_dp_sgd(privacy_param, self._sampling_rate, delta)

        # Momentum is applied by the manual buffer below (post-clip, pre-noise),
        # so the optimizer itself must not apply momentum again.
        optimizer = _get_optimizer(net, self.config, momentum=0.0)
        rng = np.random.RandomState(self._seed)

        clip_fractions: list[float] = []
        grad_norms_before: list[float] = []
        grad_norms_after: list[float] = []
        momentum_buffer: dict[str, torch.Tensor] | None = None

        for _ in range(self.config.federated.local_epochs):
            for batch in self.trainloader:
                images, labels = to_device(batch)

                per_example_grads = _compute_per_example_grads(
                    net,
                    images,
                    labels,
                    criterion,
                )

                if proximal_mu > 0:
                    # FedProx: add the deterministic drift mu * (w - w_global) to
                    # every example's gradient before clipping. The shift is the
                    # same for all examples and public, so the per-example clip
                    # still bounds each example's contribution to the release.
                    params = dict(net.named_parameters())
                    per_example_grads = {
                        k: g + proximal_mu * (params[k] - global_params[k])
                        for k, g in per_example_grads.items()
                    }

                flat_before = torch.cat(
                    [g.reshape(g.shape[0], -1) for g in per_example_grads.values()],
                    dim=1,
                )
                norms_before = torch.linalg.vector_norm(flat_before, dim=1)
                grad_norms_before.append(norms_before.detach().mean().item())

                clipped, clip_frac = _clip_per_example(per_example_grads, clip_norm)
                clip_fractions.append(clip_frac)

                avg_clipped = _average_grads(clipped)
                flat_after = torch.cat(
                    [v.reshape(1, -1) for v in avg_clipped.values()],
                    dim=1,
                )
                grad_norms_after.append(flat_after.detach().norm().item())

                momentum = self.config.optimizer.momentum
                if momentum > 0:
                    # Momentum (Opacus-style, DP-safe): applied to the averaged
                    # clipped gradient, *before* noise. Applying momentum after
                    # noise would compound the noise; applying it to per-sample
                    # gradients would require per-sample buffers. Each example
                    # appears in exactly one step with weight m^(t-i) <= 1, so
                    # the sensitivity of the buffered update stays 2C/n.
                    if momentum_buffer is None:
                        momentum_buffer = {k: v.clone() for k, v in avg_clipped.items()}
                    else:
                        for k in avg_clipped:
                            momentum_buffer[k] = momentum * momentum_buffer[k] + avg_clipped[k]
                    grad_for_noise = momentum_buffer
                else:
                    grad_for_noise = avg_clipped

                noisy_grad = _add_noise(grad_for_noise, sigma, clip_norm, rng)

                optimizer.zero_grad()
                _set_model_grads(net, noisy_grad)
                optimizer.step()

        local_weights = self.model.get_weights()

        if self._accountant is not None:
            # One communication round = one Gaussian release (spec §2): the
            # accountant steps once at the per-round sigma.
            self._accountant.step(
                sigma=sigma,
                clipping_norm=self._sampling_rate,
                num_steps=1,
                mode="per_example",
            )
            if self._rdp_native:
                cumulative_privacy = self._accountant.get_rdp_at_alpha(self._rdp_alpha)
            else:
                cumulative_privacy = self._accountant.get_epsilon()
        else:
            cumulative_privacy = 0.0

        final_flat = np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in net.parameters()],
        )
        update_norm = float(np.linalg.norm(final_flat - initial_flat))
        noisy_weights = local_weights

        utility_loss_noisy, noisy_logits = compute_validation_stats(
            self.model.get_model(),
            self.valloader,
            criterion,
        )

        if clean_pass:
            assert clean_net_ref is not None
            clean_net, update_norm_clean = _run_clean_pass(
                clean_net_ref,
                self.trainloader,
                self.config,
                criterion,
            )
            clean_net.eval()
            utility_loss_clean, clean_logits = compute_validation_stats(
                clean_net,
                self.valloader,
                criterion,
            )
        else:
            utility_loss_clean = 0.0
            update_norm_clean = 0.0
            clean_logits = None

        if clean_pass:
            loss_degradation = max(0.0, utility_loss_noisy - utility_loss_clean)
            inv_loss_clean = 1.0 / max(utility_loss_clean, 1e-12)
            utility_efficiency = -loss_degradation * inv_loss_clean / max(privacy_param, 1e-12)
            utility_retention = utility_loss_noisy * inv_loss_clean

            privacy_remaining = self._resolve_remaining_rdp()
            utility_per_remaining = (
                -loss_degradation * inv_loss_clean / max(privacy_remaining, 1e-12)
            )

            assert clean_logits is not None
            clean_flat = clean_logits.view(clean_logits.size(0), -1)
            noisy_flat_logits = noisy_logits.view(noisy_logits.size(0), -1)
            cos_sim = torch.nn.functional.cosine_similarity(
                clean_flat,
                noisy_flat_logits,
                dim=1,
            )
            # logit_disagreement = 1 - mean(cos_sim) is the minimization-equivalent
            # complement of the paper's m_agr (maximized logit agreement): minimizing
            # 1 - cos_sim maximizes cos_sim. The report schema (§6.1
            # meta.display_names) presents this metric as "agreement".
            logit_disagreement = 1.0 - cos_sim.mean().item()
        else:
            utility_efficiency = 0.0
            utility_retention = 0.0
            utility_per_remaining = 0.0
            logit_disagreement = 0.0

        mean_before = float(np.mean(grad_norms_before)) if grad_norms_before else 0.0
        mean_after = float(np.mean(grad_norms_after)) if grad_norms_after else 0.0
        # m_snr = ||Delta_clean||_2^2 / sigma^2 with the clean unclipped update
        # (spec §9.12); clean-pass methods get the clean-pass update norm,
        # others report 0.0 (clean-derived metrics are N/A for them).
        snr = (update_norm_clean**2) / max(sigma**2, 1e-12) if clean_pass else 0.0

        clipped_fraction = float(np.mean(clip_fractions)) if clip_fractions else 0.0

        if self._rdp_native:
            metrics = {
                "rdp_cost": privacy_param,
                "r_t_final": privacy_param,
                "acct_cost": compute_rdp_cost_dp_sgd(
                    self._rdp_alpha,
                    sigma,
                    self._sampling_rate,
                ),
                "cumulative_rdp": cumulative_privacy,
                "client_rdp": self._client_epsilon or 0.0,
                "update_norm": update_norm,
                "update_norm_clean": update_norm_clean,
                "utility_loss": utility_loss_noisy,
                "utility_efficiency": utility_efficiency,
                "snr": snr,
                "sigma": sigma,
                "utility_loss_clean": utility_loss_clean,
                "utility_retention": utility_retention,
                "utility_per_remaining": utility_per_remaining,
                "logit_disagreement": logit_disagreement,
                "budget_exhausted": False,
                "per_example_clip_fraction": clipped_fraction,
                "grad_norm_before_clip": mean_before,
                "grad_norm_after_clip": mean_after,
                "num_opt_steps": self._total_steps_per_round,
            }
        else:
            metrics = {
                "epsilon": privacy_param,
                "cumulative_epsilon": cumulative_privacy,
                "client_epsilon": self._client_epsilon or 0.0,
                "update_norm": update_norm,
                "update_norm_clean": update_norm_clean,
                "utility_loss": utility_loss_noisy,
                "utility_efficiency": utility_efficiency,
                "snr": snr,
                "sigma": sigma,
                "utility_loss_clean": utility_loss_clean,
                "utility_retention": utility_retention,
                "utility_per_remaining": utility_per_remaining,
                "logit_disagreement": logit_disagreement,
                "budget_exhausted": False,
                "per_example_clip_fraction": clipped_fraction,
                "grad_norm_before_clip": mean_before,
                "grad_norm_after_clip": mean_after,
                "num_opt_steps": self._total_steps_per_round,
            }

        return noisy_weights, len(self.trainloader.dataset), metrics  # type: ignore[arg-type]
