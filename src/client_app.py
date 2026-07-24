from __future__ import annotations

import logging

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.client import create_client
from src.config.loader import ExperimentConfig, load_config
from src.data import create_client_dataloader
from src.models import create_model
from src.privacy.accountant import RDPAccountant
from src.privacy.bo_scheduler import PLDPBOScheduler
from src.privacy.epsilon_scheduler import (
    EpsilonScheduler,
    FixedEpsilonScheduler,
    UniformRandomEpsilonScheduler,
)
from src.privacy.per_update_dp import enforce_epsilon_budget
from src.privacy.personalization import assign_epsilon_bounds, compute_budget_weight
from src.utils import set_seed

logger = logging.getLogger(__name__)

app = ClientApp()

ACCOUNTANT_STATE_KEY = "pldp_accountant_state"
MECHANISM_STATE_KEY = "pldp_mechanism_state"

def _prepare_metric_record(metrics: dict) -> dict:
    # Flower's MetricRecord does not accept native Python bools,
    # so they are converted to int (0/1).
    return {
        k: (int(v) if isinstance(v, bool) else v)
        for k, v in metrics.items()
    }

SCHEDULER_STATE_KEY = "pldp_scheduler_state"

_OPTIMIZATION_METRIC_KEY_MAP: dict[str, str] = {
    "nun": "update_norm",
    "utility": "utility_loss",
    "utility_efficiency": "utility_efficiency",
    "snr": "snr",
    "utility_retention": "utility_retention",
    "utility_per_remaining": "utility_per_remaining",
    "agreement": "agreement",
}


def _make_scheduler(
    partition_id: int,
    train_dataset: object,
    config: ExperimentConfig,
    _num_partitions: int,
    eps_min: float | None = None,
    eps_max: float | None = None,
    warmup_rounds: int | None = None,
    total_train_size: int | None = None,
) -> EpsilonScheduler | None:
    if not config.privacy.enabled:
        return None
    if config.bo.enabled:
        e_min = eps_min if eps_min is not None else config.bo.epsilon_min
        e_max = eps_max if eps_max is not None else config.bo.epsilon_max
        w_rounds = warmup_rounds if warmup_rounds is not None else 0
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
) -> EpsilonScheduler | None:
    if SCHEDULER_STATE_KEY in context.state:
        state = context.state[SCHEDULER_STATE_KEY]
        stype = state.get("type")
        if stype == "fixed":
            return FixedEpsilonScheduler.from_state(state)
        if stype == "uniform_random":
            return UniformRandomEpsilonScheduler.from_state(state)
        if stype == "pldp_bo":
            return PLDPBOScheduler.from_state(state)
        raise ValueError(f"Unknown scheduler type: {stype}")
    return _make_scheduler(
        partition_id, train_dataset, config, num_partitions,
        eps_min=eps_min, eps_max=eps_max, warmup_rounds=warmup_rounds,
        total_train_size=total_train_size,
    )


@app.train()
def train(msg: Message, context: Context) -> Message:
    config_path = str(context.run_config.get("config-path", "config/default.yaml"))
    overrides = {
        k: v for k, v in context.run_config.items() if k != "config-path"
    }
    config = load_config(config_path, overrides=overrides)

    if config.bo.enabled:
        _VALID_BO_METRICS = {
            "nun", "utility", "utility_efficiency", "snr",
            "utility_retention", "utility_per_remaining", "agreement",
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

    trainloader, valloader, client_subset, total_train_size = create_client_dataloader(
        config.data, partition_id, num_partitions, config.seed,
    )

    accountant: RDPAccountant | None = None
    scheduler: EpsilonScheduler | None = None
    mechanism_state: dict | None = None

    eps_min_per_client: float | None = None

    if config.privacy.enabled:
        if config.bo.enabled:
            bounds_min, bounds_max, warmup = assign_epsilon_bounds(
                partition_id, client_subset,
                config.personalization, config.bo, config.data.num_clients,
                total_train_size=total_train_size,
                num_rounds=config.federated.num_rounds,
                total_num_classes=config.model.num_classes,
            )
            eps_min_per_client = bounds_min
        else:
            bounds_min = None
            bounds_max = None
            warmup = None
            if config.privacy.target_epsilon is not None:
                eps_min_per_client = config.privacy.target_epsilon * 0.01
            elif config.privacy.total_budget is not None:
                eps_min_per_client = config.privacy.total_budget * 0.01 / config.federated.num_rounds

        if ACCOUNTANT_STATE_KEY in context.state:
            state = context.state[ACCOUNTANT_STATE_KEY]
            accountant = RDPAccountant.from_state(state)
        else:
            accountant = RDPAccountant(delta=config.privacy.delta)

        scheduler = _restore_or_create_scheduler(
            context, partition_id, client_subset, config, num_partitions,
            eps_min=bounds_min, eps_max=bounds_max, warmup_rounds=warmup,
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
    server_budget = (msg.content.config_records.get("config") or ConfigRecord()).get("per_client_budget")
    if server_budget is not None:
        total_budget = float(server_budget)

    remaining_budget: float | None = None
    if scheduler is not None and accountant is not None and total_budget is not None:
        remaining_budget = max(0.0, total_budget - accountant.get_epsilon())
        if hasattr(scheduler, "set_remaining_budget"):
            scheduler.set_remaining_budget(remaining_budget)

    if remaining_budget is not None and eps_min_per_client is not None:
        eps_min_per_client = min(eps_min_per_client, remaining_budget)

    total_steps_per_round = config.federated.local_epochs * len(trainloader)
    local_train_size = len(client_subset)

    epsilon, computed_sigma = _resolve_epsilon(
        scheduler, accountant, config, total_budget,
        eps_min=eps_min_per_client,
        total_steps_per_round=total_steps_per_round,
        local_train_size=local_train_size,
    )
    if epsilon < 0:
        logger.info(
            "Client %d privacy budget exhausted (epsilon=%.4f), ceasing participation",
            partition_id,
            epsilon,
        )
        epsilon = 0.0  # sentinel: PerUpdateDPClient._check_budget checks for == 0
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
    )

    arrays_raw = msg.content.get("arrays")
    if not isinstance(arrays_raw, ArrayRecord):
        raise TypeError(f"Expected ArrayRecord, got {type(arrays_raw).__name__}")
    arrays: ArrayRecord = arrays_raw
    parameters = arrays.to_numpy_ndarrays()
    num_examples, fit_metrics = client.fit(parameters, {})[1:]

    if config.bo.enabled and scheduler is not None and accountant is not None and not fit_metrics.get("budget_exhausted", False):
        metric_key = _OPTIMIZATION_METRIC_KEY_MAP[config.bo.optimization_metric]
        metric_value = fit_metrics.get(metric_key)
        if metric_value is not None:
            scheduler.step(epsilon, float(metric_value))

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
    metric_record = MetricRecord(_prepare_metric_record(metrics))
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


def _resolve_epsilon(
    scheduler: EpsilonScheduler | None,
    accountant: RDPAccountant | None,
    config: ExperimentConfig,
    total_budget: float | None = None,
    eps_min: float | None = None,
    total_steps_per_round: int | None = None,
    local_train_size: int | None = None,
) -> tuple[float, float]:
    """Return (epsilon, sigma) where sigma is None if no budget enforcement applies."""
    if scheduler is not None:
        candidate = scheduler.get_epsilon()
    elif config.personalization.enabled:
        # Per-round epsilon = per_client_budget / total_rounds.
        # Without a per-client budget (total_budget is None or 0) the client
        # cannot derive a meaningful epsilon and drops out. This is intentional:
        # the old assign_epsilon fallback (fixed epsilon with no budget cap) is
        # removed — personalization without total_budget is a misconfiguration.
        if total_budget is not None and total_budget > 0:
            candidate = total_budget / config.federated.num_rounds
        else:
            return 0.0, 0.0
    elif config.privacy.target_epsilon is not None:
        candidate = config.privacy.target_epsilon
    elif config.privacy.enabled:
        raise ValueError(
            "Privacy enabled but no epsilon source available. "
            "Set privacy.target_epsilon, enable a scheduler (bo/personalization), "
            "or disable privacy."
        )
    else:
        return 0.0, 0.0

    if accountant is not None and total_budget is not None:
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
            num_steps = total_steps_per_round if total_steps_per_round is not None else 1
            candidate, computed_sigma = enforce_epsilon_budget(
                candidate, accountant.rdp_per_alpha, total_budget,
                lower_bound, sampling_rate, delta,
                clipping_mode="per_example", num_steps=num_steps,
            )
        else:
            c = config.privacy.update_clip_norm
            candidate, computed_sigma = enforce_epsilon_budget(
                candidate, accountant.rdp_per_alpha, total_budget,
                lower_bound, c, delta,
            )
        return candidate, computed_sigma

    return candidate, 0.0


@app.query()
def query(msg: Message, context: Context) -> Message:
    config_path = str(context.run_config.get("config-path", "config/default.yaml"))
    overrides = {
        k: v for k, v in context.run_config.items() if k != "config-path"
    }
    config = load_config(config_path, overrides=overrides)

    task = (msg.content.config_records.get("config") or ConfigRecord()).get("task")
    if task == "personalization_metadata":
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        _, _, client_subset, total_train_size = create_client_dataloader(
            config.data, partition_id, num_partitions, config.seed,
        )

        weight = compute_budget_weight(
            partition_id, client_subset, config.personalization,
            num_clients=config.data.num_clients,
            total_train_size=total_train_size,
            rng=np.random.RandomState(config.seed + partition_id),
            total_num_classes=config.model.num_classes,
        )

        return Message(
            content=RecordDict({
                "config": ConfigRecord({
                    "partition_id": partition_id,
                    "budget_weight": weight,
                }),
            }),
            reply_to=msg,
        )

    raise ValueError(f"Unknown query task: {task}")


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    config_path = str(context.run_config.get("config-path", "config/default.yaml"))
    overrides = {
        k: v for k, v in context.run_config.items() if k != "config-path"
    }
    config = load_config(config_path, overrides=overrides)


    set_seed(config.seed, deterministic=config.deterministic)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    trainloader, valloader, *_ = create_client_dataloader(
        config.data, partition_id, num_partitions, config.seed,
    )

    client_epsilon = None
    accountant = None
    mechanism_state = None
    if config.privacy.enabled and ACCOUNTANT_STATE_KEY in context.state:
            state = context.state[ACCOUNTANT_STATE_KEY]
            accountant = RDPAccountant.from_state(state)
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
    metrics["cumulative_epsilon"] = client_epsilon if client_epsilon is not None else 0.0
    metric_record = MetricRecord(_prepare_metric_record(metrics))
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
