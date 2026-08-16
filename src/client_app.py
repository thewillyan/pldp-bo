from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Union, cast

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch.utils.data import Subset

from src.client import create_client
from src.config.loader import ExperimentConfig, load_config
from src.config.locked import FIXED_METHODS, assert_locked_config
from src.data import create_client_dataloader, create_dataset, create_test_dataset
from src.data.dataloaders import create_dataloaders
from src.device import get_device, to_device
from src.models import create_model
from src.privacy.accountant import RDPAccountant
from src.privacy.bo_scheduler import PLDPBORDPScheduler, PLDPBOScheduler
from src.privacy.epsilon_scheduler import (
    EpsilonScheduler,
    FixedEpsilonScheduler,
    FixedRDPScheduler,
    RDPNativeScheduler,
    UniformRandomEpsilonScheduler,
    UniformRandomRDPScheduler,
)
from src.privacy.per_update_dp import (
    _sigma_for_rdp_target,
    _sigma_for_rdp_target_dp_sgd,
    enforce_epsilon_budget,
    enforce_rdp_budget,
)
from src.privacy.personalization import assign_epsilon_bounds, compute_budget_weight
from src.utils import set_seed

logger = logging.getLogger(__name__)

app = ClientApp()

ACCOUNTANT_STATE_KEY = "pldp_accountant_state"
MECHANISM_STATE_KEY = "pldp_mechanism_state"

AnyScheduler = Union[EpsilonScheduler, RDPNativeScheduler]


def _prepare_metric_record(metrics: dict) -> dict:
    # Flower's MetricRecord does not accept native Python bools,
    # so they are converted to int (0/1).
    return {k: (int(v) if isinstance(v, bool) else v) for k, v in metrics.items()}


SCHEDULER_STATE_KEY = "pldp_scheduler_state"

_OPTIMIZATION_METRIC_KEY_MAP: dict[str, str] = {
    "nun": "update_norm",
    "utility": "utility_loss",
    "utility_efficiency": "utility_efficiency",
    "snr": "snr",
    "utility_retention": "utility_retention",
    "utility_per_remaining": "utility_per_remaining",
    "logit_disagreement": "logit_disagreement",
}


def _is_rdp_native(config: ExperimentConfig) -> bool:
    return config.privacy.accountant_mode == "rdp_native"


def _make_scheduler(
    partition_id: int,
    _train_dataset: object,
    config: ExperimentConfig,
    _num_partitions: int,
    eps_min: float | None = None,
    eps_max: float | None = None,
    warmup_rounds: int | None = None,
    _total_train_size: int | None = None,
) -> AnyScheduler | None:
    if not config.privacy.enabled:
        return None

    if _is_rdp_native(config):
        return _make_rdp_native_scheduler(
            partition_id,
            config,
            eps_min,
            eps_max,
            warmup_rounds,
        )

    if config.bo.enabled:
        e_min = eps_min if eps_min is not None else config.bo.epsilon_min
        e_max = eps_max if eps_max is not None else config.bo.epsilon_max
        w_rounds = warmup_rounds if warmup_rounds is not None else config.bo.min_warmup
        return PLDPBOScheduler(
            epsilon_min=e_min,
            epsilon_max=e_max,
            warmup_rounds=w_rounds,
            acquisition_penalty=config.bo.acquisition_penalty,
            grid_points=config.bo.grid_points,
            gp_kernel=config.bo.gp_kernel,
            observation_noise=config.bo.observation_noise,
            budget_margin=config.bo.budget_margin,
            ema_alpha=config.bo.ema_alpha,
            seed=config.seed + partition_id,
        )
    if config.personalization.enabled:
        return None
    if config.privacy.target_epsilon is not None:
        return FixedEpsilonScheduler(config.privacy.target_epsilon)
    return UniformRandomEpsilonScheduler(
        epsilon_min=config.bo.epsilon_min,
        epsilon_max=config.bo.epsilon_max,
        seed=config.seed + partition_id,
    )


def _make_rdp_native_scheduler(
    partition_id: int,
    config: ExperimentConfig,
    rdp_min: float | None = None,
    rdp_max: float | None = None,
    warmup_rounds: int | None = None,
) -> RDPNativeScheduler | None:
    """Create an RDP-native scheduler."""
    if config.bo.enabled:
        r_min = rdp_min if rdp_min is not None else config.bo.rdp_min
        r_max = rdp_max if rdp_max is not None else config.bo.rdp_max
        w_rounds = warmup_rounds if warmup_rounds is not None else config.bo.min_warmup
        return PLDPBORDPScheduler(
            rdp_min=r_min,
            rdp_max=r_max,
            warmup_rounds=w_rounds,
            num_rounds=config.federated.num_rounds,
            acquisition_penalty=config.bo.acquisition_penalty,
            grid_points=config.bo.grid_points,
            gp_kernel=config.bo.gp_kernel,
            observation_noise=config.bo.observation_noise,
            budget_margin=config.bo.budget_margin,
            ema_alpha=config.bo.ema_alpha,
            seed=config.seed + partition_id,
        )
    if config.method in FIXED_METHODS:
        # Fixed baselines (§9.5): R = B_RDP/(rho*T) = 0.5 per round, from the
        # locked privacy.fixed_rdp_target.
        return FixedRDPScheduler(rdp_target=config.privacy.fixed_rdp_target)
    if config.personalization.enabled:
        return None
    # For non-BO RDP-native mode, use uniform random over rdp range
    return UniformRandomRDPScheduler(
        rdp_min=config.bo.rdp_min,
        rdp_max=config.bo.rdp_max,
        seed=config.seed + partition_id,
    )


def _read_remaining_rdp(msg: Message) -> float | None:
    """Read the server-sent remaining RDP budget (B_RDP - cum_rdp) from a fit message.

    Returns None when absent — the DP clients then raise (IMPL-10: the
    server must send ``remaining_rdp`` each round; there is no fallback).
    """
    raw = (msg.content.config_records.get("config") or ConfigRecord()).get("remaining_rdp")
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _restore_or_create_scheduler(
    context: Context,
    partition_id: int,
    train_dataset: object,
    config: ExperimentConfig,
    num_partitions: int,
    eps_min: float | None = None,
    eps_max: float | None = None,
    warmup_rounds: int | None = None,
    total_train_size: int | None = None,
) -> AnyScheduler | None:
    if SCHEDULER_STATE_KEY in context.state:
        state = context.state[SCHEDULER_STATE_KEY]
        stype = state.get("type")
        # Epsilon-based schedulers
        if stype == "fixed":
            return FixedEpsilonScheduler.from_state(state)
        if stype == "uniform_random":
            return UniformRandomEpsilonScheduler.from_state(state)
        if stype == "pldp_bo":
            return PLDPBOScheduler.from_state(state)
        # RDP-native schedulers
        if stype == "fixed_rdp":
            return FixedRDPScheduler.from_state(state)
        if stype == "uniform_random_rdp":
            return UniformRandomRDPScheduler.from_state(state)
        if stype == "pldp_bo_rdp":
            return PLDPBORDPScheduler.from_state(state)
        raise ValueError(f"Unknown scheduler type: {stype}")
    return _make_scheduler(
        partition_id,
        train_dataset,
        config,
        num_partitions,
        eps_min=eps_min,
        eps_max=eps_max,
        warmup_rounds=warmup_rounds,
        _total_train_size=total_train_size,
    )


@app.train()
def train(msg: Message, context: Context) -> Message:
    config_path = str(context.run_config.get("config-path", "config/default.yaml"))
    overrides = {
        k: v
        for k, v in context.run_config.items()
        if k not in ("config-path", "app_config_overrides")
    }
    config = load_config(config_path, overrides=overrides)

    assert_locked_config(config)

    if config.bo.enabled:
        _VALID_BO_METRICS = {
            "nun",
            "utility",
            "utility_efficiency",
            "snr",
            "utility_retention",
            "utility_per_remaining",
            "logit_disagreement",
        }
        if config.bo.optimization_metric not in _VALID_BO_METRICS:
            raise ValueError(
                f"Invalid bo.optimization_metric='{config.bo.optimization_metric}'. "
                f"Must be one of {sorted(_VALID_BO_METRICS)}."
            )

    set_seed(config.seed, deterministic=config.deterministic)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    client_seed = config.seed + partition_id

    trainloader, valloader, client_subset, _, total_train_size = create_client_dataloader(
        config.data,
        partition_id,
        num_partitions,
        config.seed,
    )

    accountant: RDPAccountant | None = None
    scheduler: AnyScheduler | None = None
    mechanism_state: dict | None = None

    rdp_native = _is_rdp_native(config)

    # Bounds for scheduler search space
    bounds_min: float | None = None
    bounds_max: float | None = None
    warmup: int | None = None

    if config.privacy.enabled:
        if config.bo.enabled:
            if rdp_native:
                if config.bo.bounds_strategy == "from_rdp":
                    bounds_min, bounds_max, warmup = assign_epsilon_bounds(
                        partition_id,
                        client_subset,
                        config.personalization,
                        config.bo,
                        config.data.num_clients,
                        total_train_size=total_train_size,
                        num_rounds=config.federated.num_rounds,
                        total_num_classes=config.model.num_classes,
                    )
                else:
                    bounds_min = config.bo.rdp_min
                    bounds_max = config.bo.rdp_max
                    warmup = config.bo.min_warmup
            else:
                bounds_min, bounds_max, warmup = assign_epsilon_bounds(
                    partition_id,
                    client_subset,
                    config.personalization,
                    config.bo,
                    config.data.num_clients,
                    total_train_size=total_train_size,
                    num_rounds=config.federated.num_rounds,
                    total_num_classes=config.model.num_classes,
                )

        if ACCOUNTANT_STATE_KEY in context.state:
            state = context.state[ACCOUNTANT_STATE_KEY]
            accountant = RDPAccountant.from_state(state)
        else:
            accountant = RDPAccountant(delta=config.privacy.delta)

        scheduler = _restore_or_create_scheduler(
            context,
            partition_id,
            client_subset,
            config,
            num_partitions,
            eps_min=bounds_min,
            eps_max=bounds_max,
            warmup_rounds=warmup,
            total_train_size=total_train_size,
        )
        if scheduler is not None:
            logger.info(
                "Client %d scheduler: %s",
                partition_id,
                scheduler,
            )

        if MECHANISM_STATE_KEY in context.state:
            mechanism_state = context.state[MECHANISM_STATE_KEY]

    total_budget: float | None = None
    if config.privacy.total_budget is not None:
        total_budget = config.privacy.total_budget
    elif config.bo.enabled:
        total_budget = config.bo.epsilon_budget

    # Server-assigned per-client budget overrides the config-derived value
    server_budget = (msg.content.config_records.get("config") or ConfigRecord()).get(
        "per_client_budget"
    )
    if server_budget is not None:
        total_budget = float(server_budget)

    # Server-sent remaining RDP budget (B_RDP - cum_rdp) for m_rem (IMPL-10);
    # required by the DP clients — a missing value raises client-side.
    server_remaining_rdp = _read_remaining_rdp(msg)

    remaining_budget: float | None = None
    if scheduler is not None and accountant is not None and total_budget is not None:
        if rdp_native:
            alpha = config.privacy.rdp_alpha
            remaining_budget = max(0.0, total_budget - accountant.get_rdp_at_alpha(alpha))
        else:
            remaining_budget = max(0.0, total_budget - accountant.get_epsilon())
        if hasattr(scheduler, "set_remaining_budget"):
            scheduler.set_remaining_budget(remaining_budget)

    # Resolve the privacy parameter (epsilon or rdp cost)
    r_t_candidate: float = 0.0
    if rdp_native:
        rdp_cost, computed_sigma, r_t_candidate, bo_time, acct_time = _resolve_rdp(
            scheduler,
            accountant,
            config,
            total_budget,
            local_train_size=len(client_subset),
        )
        if rdp_cost < 0:
            logger.info(
                "Client %d privacy budget exhausted (rdp_cost=%.6f), ceasing participation",
                partition_id,
                rdp_cost,
            )
            rdp_cost = 0.0
            computed_sigma = 0.0
        logger.debug("Client %d using rdp_cost=%.6f", partition_id, rdp_cost)

        # Pass rdp_cost as client_epsilon — the client uses it as a generic
        # "privacy parameter" and the computed_sigma is the actual noise scale.
        client_model = create_model(config.model, dataset_name=config.data.name)
        client = create_client(
            cid=partition_id,
            model=client_model,
            trainloader=trainloader,
            valloader=valloader,
            config=config,
            client_epsilon=rdp_cost,
            computed_sigma=computed_sigma if computed_sigma > 0 else None,
            accountant=accountant,
            seed=client_seed,
            mechanism_state=mechanism_state,
            remaining_budget=remaining_budget,
            remaining_rdp=server_remaining_rdp,
        )
    else:
        epsilon, computed_sigma, eps_candidate, bo_time, acct_time = _resolve_epsilon(
            scheduler,
            accountant,
            config,
            total_budget,
            local_train_size=len(client_subset),
        )
        if epsilon < 0:
            logger.info(
                "Client %d privacy budget exhausted (epsilon=%.4f), ceasing participation",
                partition_id,
                epsilon,
            )
            epsilon = 0.0
            computed_sigma = 0.0
        logger.debug("Client %d using epsilon=%.4f", partition_id, epsilon)

        client_model = create_model(config.model, dataset_name=config.data.name)
        client = create_client(
            cid=partition_id,
            model=client_model,
            trainloader=trainloader,
            valloader=valloader,
            config=config,
            client_epsilon=epsilon,
            computed_sigma=computed_sigma if computed_sigma > 0 else None,
            accountant=accountant,
            seed=client_seed,
            mechanism_state=mechanism_state,
            remaining_budget=remaining_budget,
            remaining_rdp=server_remaining_rdp,
        )

    arrays_raw = msg.content.get("arrays")
    if not isinstance(arrays_raw, ArrayRecord):
        raise TypeError(f"Expected ArrayRecord, got {type(arrays_raw).__name__}")
    arrays: ArrayRecord = arrays_raw
    parameters = arrays.to_numpy_ndarrays()
    num_examples, fit_metrics = client.fit(parameters, {})[1:]

    if (
        config.bo.enabled
        and scheduler is not None
        and accountant is not None
        and not fit_metrics.get("budget_exhausted", False)
    ):
        metric_key = _OPTIMIZATION_METRIC_KEY_MAP[config.bo.optimization_metric]
        metric_value = fit_metrics.get(metric_key)
        if metric_value is not None:
            if rdp_native:
                rdp_round = fit_metrics.get("rdp_cost", 0.0)
                # Both per_update and per_example clients report per-round
                # RDP cost, matching the scheduler's grid domain.
                scheduler.step(rdp_round, float(metric_value))
            else:
                epsilon_val = fit_metrics.get("epsilon", 0.0)
                scheduler.step(epsilon_val, float(metric_value))

    if accountant is not None and config.privacy.enabled:
        context.state[ACCOUNTANT_STATE_KEY] = ConfigRecord(accountant.get_state())

    if scheduler is not None and config.privacy.enabled:
        context.state[SCHEDULER_STATE_KEY] = ConfigRecord(scheduler.get_state())

    if config.privacy.enabled and config.privacy.clipping_mode != "per_example":
        context.state[MECHANISM_STATE_KEY] = ConfigRecord(client.get_mechanism_state())  # type: ignore[union-attr]

    model_record = ArrayRecord(client_model.get_model().state_dict())
    metrics = {
        "num-examples": num_examples,
        "client-id": partition_id,
        **fit_metrics,
    }
    if config.privacy.enabled:
        # §4.4 client_state fields: phase, observed_m, r_t_candidate and the
        # BO/accounting wall times (§4.3 bo_time_round / acct_time_round).
        exhausted = bool(fit_metrics.get("budget_exhausted", False))
        metrics["phase"] = phase_code(
            "exhausted" if exhausted else _scheduler_phase(scheduler),
        )
        metrics["bo_time"] = bo_time
        metrics["acct_time"] = acct_time
        if rdp_native:
            metrics["r_t_candidate"] = r_t_candidate
        if config.bo.enabled and not exhausted:
            metric_key = _OPTIMIZATION_METRIC_KEY_MAP[config.bo.optimization_metric]
            observed = fit_metrics.get(metric_key)
            if observed is not None:
                metrics["observed_m"] = float(observed)
    metric_record = MetricRecord(_prepare_metric_record(metrics))
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


def _scheduler_phase(scheduler: AnyScheduler | None) -> str:
    """BO phase of the scheduler ('warmup' | 'bo'); fixed/uniform → 'bo'.

    Only the PLDPBO schedulers track a warm-up/BO phase; the fixed and
    uniform-random schedulers have no warm-up, so their phase is 'bo'
    throughout (§4.4 fixed-baseline constants).
    """
    phase = getattr(scheduler, "_phase", None)
    return phase if isinstance(phase, str) else "bo"


# MetricRecord only accepts numeric values, so the client encodes the §4.4
# ``phase`` string as a numeric code; the server decodes it when building the
# client_state artifact.
_PHASE_CODES: dict[str, float] = {"warmup": 0.0, "bo": 1.0, "exhausted": 2.0}


def phase_code(phase: str) -> float:
    """Numeric encoding of the §4.4 phase string for the client reply."""
    return _PHASE_CODES[phase]


def _resolve_epsilon(
    scheduler: EpsilonScheduler | None,
    accountant: RDPAccountant | None,
    config: ExperimentConfig,
    total_budget: float | None = None,
    eps_min: float | None = None,
    local_train_size: int | None = None,
) -> tuple[float, float, float, float, float]:
    """Return (epsilon, sigma, eps_candidate, bo_time, acct_time)."""
    bo_time = 0.0
    if scheduler is not None:
        if getattr(scheduler, "_phase", None) is not None:
            t0 = perf_counter()
            candidate = scheduler.get_epsilon()
            bo_time = perf_counter() - t0
        else:
            candidate = scheduler.get_epsilon()
    elif config.privacy.target_epsilon is not None:
        candidate = config.privacy.target_epsilon
    elif config.privacy.enabled:
        raise ValueError(
            "Privacy enabled but no epsilon source available. "
            "Set privacy.target_epsilon, enable a scheduler (bo), "
            "or disable privacy."
        )
    else:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    eps_candidate = candidate

    if accountant is not None and total_budget is not None:
        t0 = perf_counter()
        delta = config.privacy.delta
        lower_bound = eps_min if eps_min is not None else 1e-6
        clipping_mode = config.privacy.clipping_mode

        if clipping_mode == "per_example":
            if local_train_size is None:
                raise ValueError(
                    "clipping_mode='per_example' requires local_train_size "
                    "to compute sampling rate for budget enforcement."
                )
            sampling_rate = config.data.batch_size / local_train_size
            # One communication round = one Gaussian release: keep the epsilon
            # path in sync with the per-round RDP accounting convention.
            candidate, computed_sigma = enforce_epsilon_budget(
                candidate,
                accountant.rdp_per_alpha,
                total_budget,
                lower_bound,
                0.0,
                delta,
                clipping_mode="per_example",
                num_steps=1,
                sampling_rate=sampling_rate,
            )
        else:
            c = config.privacy.update_clip_norm
            candidate, computed_sigma = enforce_epsilon_budget(
                candidate,
                accountant.rdp_per_alpha,
                total_budget,
                lower_bound,
                c,
                delta,
            )
        acct_time = perf_counter() - t0
        return candidate, computed_sigma, eps_candidate, bo_time, acct_time

    return candidate, 0.0, eps_candidate, bo_time, 0.0


def _resolve_rdp(
    scheduler: RDPNativeScheduler | None,
    accountant: RDPAccountant | None,
    config: ExperimentConfig,
    total_budget: float | None = None,
    eps_min: float | None = None,
    local_train_size: int | None = None,
) -> tuple[float, float, float, float, float]:
    """Return (rdp_cost, sigma, r_t_candidate, bo_time, acct_time) for RDP-native mode.

    ``r_t_candidate`` is the scheduler's proposed per-round RDP cost before
    budget enforcement; ``bo_time`` is the wall time of the BO decision (GP fit
    + acquisition + selection, §4.3) and ``acct_time`` the wall time of the
    budget check + sigma calibration.
    """
    alpha = config.privacy.rdp_alpha

    bo_time = 0.0
    if scheduler is not None:
        if getattr(scheduler, "_phase", None) is not None:
            t0 = perf_counter()
            candidate = scheduler.get_rdp()
            bo_time = perf_counter() - t0
        else:
            candidate = scheduler.get_rdp()
    else:
        raise ValueError(
            "RDP-native mode requires a scheduler (bo) or personalization with total_budget."
        )

    r_t_candidate = candidate

    if accountant is not None and total_budget is not None:
        t0 = perf_counter()
        current_rdp = accountant.get_rdp_at_alpha(alpha)
        lower_bound = eps_min if eps_min is not None else 1e-6
        clipping_mode = config.privacy.clipping_mode

        if clipping_mode == "per_example":
            if local_train_size is None:
                raise ValueError("clipping_mode='per_example' requires local_train_size.")
            sampling_rate = config.data.batch_size / local_train_size
            # One communication round = one Gaussian release (spec §2): the
            # scheduler's candidate is the per-round RDP cost R_t and sigma is
            # calibrated per-round, sigma_t = sqrt(alpha * q^2 / (2 * R_t)).
            candidate, computed_sigma = enforce_rdp_budget(
                candidate,
                current_rdp,
                total_budget,
                lower_bound,
                alpha,
                config.privacy.update_clip_norm,
                clipping_mode="per_example",
                num_steps=1,
                sampling_rate=sampling_rate,
            )
        else:
            c = config.privacy.update_clip_norm
            candidate, computed_sigma = enforce_rdp_budget(
                candidate,
                current_rdp,
                total_budget,
                lower_bound,
                alpha,
                c,
            )
        acct_time = perf_counter() - t0
        return candidate, computed_sigma, r_t_candidate, bo_time, acct_time

    # No budget enforcement — compute sigma directly from RDP cost
    t0 = perf_counter()
    clipping_mode = config.privacy.clipping_mode
    if clipping_mode == "per_example":
        if local_train_size is not None:
            sampling_rate = config.data.batch_size / local_train_size
            sigma = _sigma_for_rdp_target_dp_sgd(candidate, alpha, sampling_rate)
        else:
            sigma = 0.0
    else:
        c = config.privacy.update_clip_norm
        sigma = _sigma_for_rdp_target(candidate, alpha, c)
    acct_time = perf_counter() - t0
    return candidate, sigma, r_t_candidate, bo_time, acct_time


@app.query()
def query(msg: Message, context: Context) -> Message:
    config_path = str(context.run_config.get("config-path", "config/default.yaml"))
    overrides = {
        k: v
        for k, v in context.run_config.items()
        if k not in ("config-path", "app_config_overrides")
    }
    config = load_config(config_path, overrides=overrides)

    task = (msg.content.config_records.get("config") or ConfigRecord()).get("task")
    if task == "personalization_metadata":
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        _, _, client_subset, _, total_train_size = create_client_dataloader(
            config.data,
            partition_id,
            num_partitions,
            config.seed,
        )

        weight = compute_budget_weight(
            partition_id,
            client_subset,
            config.personalization,
            num_clients=config.data.num_clients,
            total_train_size=total_train_size,
            rng=np.random.RandomState(config.seed + partition_id),
            total_num_classes=config.model.num_classes,
        )

        return Message(
            content=RecordDict(
                {
                    "config": ConfigRecord(
                        {
                            "partition_id": partition_id,
                            "budget_weight": weight,
                        }
                    ),
                }
            ),
            reply_to=msg,
        )

    if task == "client_test_accuracy":
        if config.data.name != "femnist":
            raise ValueError("client_test_accuracy requires the femnist dataset")
        arrays_raw = msg.content.get("arrays")
        if not isinstance(arrays_raw, ArrayRecord):
            raise TypeError(f"Expected ArrayRecord, got {type(arrays_raw).__name__}")
        arrays: ArrayRecord = arrays_raw

        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])

        _, _, client_subset, _, _ = create_client_dataloader(
            config.data,
            partition_id,
            num_partitions,
            config.seed,
        )
        train_dataset = create_dataset(config.data)
        writer_set = set(
            cast(Any, train_dataset).users[cast(Any, client_subset).indices].tolist(),
        )

        test_dataset = create_test_dataset(config.data)
        test_idx = [
            i for i, u in enumerate(cast(Any, test_dataset).users.tolist()) if u in writer_set
        ]

        client_model = create_model(config.model, dataset_name=config.data.name)
        client_model.set_weights(arrays.to_numpy_ndarrays())
        net = client_model.get_model().to(get_device())
        net.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            test_loader = create_dataloaders(
                Subset(test_dataset, test_idx),
                config.data.batch_size,
                shuffle=False,
            )
            for batch_images, batch_labels in test_loader:
                images, labels = to_device((batch_images, batch_labels))
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += int((predicted == labels).sum().item())

        test_accuracy = correct / total if total > 0 else 0.0
        return Message(
            content=RecordDict(
                {
                    "config": ConfigRecord(
                        {
                            "partition_id": partition_id,
                            "test_accuracy": test_accuracy,
                            "n_test": total,
                        }
                    ),
                }
            ),
            reply_to=msg,
        )

    raise ValueError(f"Unknown query task: {task}")


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    config_path = str(context.run_config.get("config-path", "config/default.yaml"))
    overrides = {
        k: v
        for k, v in context.run_config.items()
        if k not in ("config-path", "app_config_overrides")
    }
    config = load_config(config_path, overrides=overrides)

    set_seed(config.seed, deterministic=config.deterministic)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    trainloader, valloader, *_ = create_client_dataloader(
        config.data,
        partition_id,
        num_partitions,
        config.seed,
    )

    client_epsilon = None
    accountant = None
    mechanism_state = None
    rdp_native = _is_rdp_native(config)

    if config.privacy.enabled and ACCOUNTANT_STATE_KEY in context.state:
        state = context.state[ACCOUNTANT_STATE_KEY]
        accountant = RDPAccountant.from_state(state)
        if rdp_native:
            client_epsilon = accountant.get_rdp_at_alpha(config.privacy.rdp_alpha)
        else:
            client_epsilon = accountant.get_epsilon()
    if config.privacy.enabled and MECHANISM_STATE_KEY in context.state:
        mechanism_state = context.state[MECHANISM_STATE_KEY]

    client_model = create_model(config.model, dataset_name=config.data.name)
    client = create_client(
        cid=partition_id,
        model=client_model,
        trainloader=trainloader,
        valloader=valloader,
        config=config,
        client_epsilon=client_epsilon,
        accountant=accountant,
        mechanism_state=mechanism_state,
    )

    arrays_raw = msg.content.get("arrays")
    if not isinstance(arrays_raw, ArrayRecord):
        raise TypeError(f"Expected ArrayRecord, got {type(arrays_raw).__name__}")
    arrays: ArrayRecord = arrays_raw
    parameters = arrays.to_numpy_ndarrays()
    loss, num_examples, eval_metrics = client.evaluate(parameters, {})

    metrics = {
        "loss": loss,
        "num-examples": num_examples,
        "client-id": partition_id,
        **eval_metrics,
    }
    if rdp_native:
        metrics["cumulative_rdp"] = client_epsilon if client_epsilon is not None else 0.0
    else:
        metrics["cumulative_epsilon"] = client_epsilon if client_epsilon is not None else 0.0
    metric_record = MetricRecord(_prepare_metric_record(metrics))
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
