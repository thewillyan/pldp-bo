from __future__ import annotations

import logging

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
from src.privacy.personalization import assign_epsilon, assign_epsilon_bounds
from src.utils import set_seed

logger = logging.getLogger(__name__)

app = ClientApp()

ACCOUNTANT_STATE_KEY = "pldp_accountant_state"

def _sanitize_metrics(metrics: dict) -> dict:
    return {
        k: (int(v) if isinstance(v, bool) else v)
        for k, v in metrics.items()
    }



SCHEDULER_STATE_KEY = "pldp_scheduler_state"

_OPTIMIZATION_METRIC_KEY_MAP: dict[str, str] = {
    "nun": "update_norm",
    "utility": "utility_loss",
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
        w_rounds = warmup_rounds if warmup_rounds is not None else config.bo.warmup_rounds
        return PLDPBOScheduler(
            epsilon_min=e_min,
            epsilon_max=e_max,
            warmup_rounds=w_rounds,
            acquisition_penalty=config.bo.acquisition_penalty,
            grid_points=config.bo.grid_points,
            gp_kernel=config.bo.gp_kernel,
            observation_noise=config.bo.observation_noise,
            seed=config.seed,
        )
    if config.personalization.enabled:
        epsilon = assign_epsilon(
            partition_id,
            train_dataset,
            config.personalization,
            num_clients=config.data.num_clients,
            total_train_size=total_train_size,
        )
        return FixedEpsilonScheduler(epsilon)
    if config.privacy.target_epsilon is not None:
        return FixedEpsilonScheduler(config.privacy.target_epsilon)
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


    set_seed(config.seed, deterministic=config.deterministic)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    trainloader, valloader, client_subset, total_train_size = create_client_dataloader(
        config.data, partition_id, num_partitions, config.seed,
    )

    accountant: RDPAccountant | None = None
    scheduler: EpsilonScheduler | None = None

    eps_min_per_client: float | None = None
    if config.bo.enabled and config.privacy.enabled:
        bounds_min, bounds_max, warmup = assign_epsilon_bounds(
            partition_id, client_subset,
            config.personalization, config.bo, config.data.num_clients,
            total_train_size=total_train_size,
        )
        eps_min_per_client = bounds_min
    else:
        bounds_min = config.bo.epsilon_min
        bounds_max = config.bo.epsilon_max
        warmup = config.bo.warmup_rounds

    if config.privacy.enabled:
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

    total_budget: float | None = None
    if config.privacy.total_budget is not None:
        total_budget = config.privacy.total_budget
    elif config.bo.enabled:
        total_budget = config.bo.epsilon_budget

    epsilon = _resolve_epsilon(
        scheduler, accountant, config, total_budget,
        eps_min=eps_min_per_client,
    )
    if epsilon < 0:
        logger.info(
            "Client %d privacy budget exhausted (epsilon=%.4f), ceasing participation",
            partition_id,
            epsilon,
        )
        epsilon = 0.0
    logger.debug("Client %d using epsilon=%.4f", partition_id, epsilon)

    client_model = create_model(config.model, dataset_name=config.data.name)
    client = create_client(
        cid=partition_id,
        model=client_model,
        trainloader=trainloader,
        valloader=valloader,
        config=config,
        client_epsilon=epsilon,
        accountant=accountant,
        total_budget=total_budget,
    )

    arrays_raw = msg.content.get("arrays")
    if not isinstance(arrays_raw, ArrayRecord):
        raise TypeError(f"Expected ArrayRecord, got {type(arrays_raw).__name__}")
    arrays: ArrayRecord = arrays_raw
    parameters = arrays.to_numpy_ndarrays()
    num_examples, fit_metrics = client.fit(parameters, {})[1:]

    if scheduler is not None and accountant is not None and not fit_metrics.get("budget_exhausted", False):
            metric_key = _OPTIMIZATION_METRIC_KEY_MAP.get(
                config.bo.optimization_metric, config.bo.optimization_metric,
            )
            metric_value = fit_metrics.get(metric_key)
            if metric_value is not None:
                scheduler.step(epsilon, float(metric_value))

    if accountant is not None and config.privacy.enabled:
        context.state[ACCOUNTANT_STATE_KEY] = ConfigRecord(accountant.get_state())

    if scheduler is not None and config.privacy.enabled:
        context.state[SCHEDULER_STATE_KEY] = ConfigRecord(scheduler.get_state())

    model_record = ArrayRecord(client_model.get_model().state_dict())
    metrics = {
        "num-examples": num_examples,
        "client-id": partition_id,
        **fit_metrics,
    }
    metric_record = MetricRecord(_sanitize_metrics(metrics))
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


def _resolve_epsilon(
    scheduler: EpsilonScheduler | None,
    accountant: RDPAccountant | None,
    config: ExperimentConfig,
    total_budget: float | None = None,
    eps_min: float | None = None,
) -> float:
    if scheduler is not None:
        candidate = scheduler.get_epsilon()
    elif config.privacy.target_epsilon is not None:
        candidate = config.privacy.target_epsilon
    else:
        return config.privacy.noise_multiplier

    if accountant is not None and total_budget is not None:
        c = config.privacy.max_grad_norm
        delta = config.privacy.delta
        lower_bound = eps_min if eps_min is not None else config.bo.epsilon_min
        candidate = enforce_epsilon_budget(
            candidate, accountant.rdp_per_alpha, total_budget,
            lower_bound, c, delta,
        )

    return candidate


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
    if config.privacy.enabled and config.personalization.enabled and ACCOUNTANT_STATE_KEY in context.state:
            state = context.state[ACCOUNTANT_STATE_KEY]
            accountant = RDPAccountant.from_state(state)
            client_epsilon = accountant.get_epsilon()

    client_model = create_model(config.model, dataset_name=config.data.name)
    client = create_client(
        cid=partition_id,
        model=client_model,
        trainloader=trainloader,
        valloader=valloader,
        config=config,
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
    metric_record = MetricRecord(_sanitize_metrics(metrics))
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
