from __future__ import annotations

import logging

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.client import create_client
from src.config.loader import load_config
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
from src.privacy.personalization import assign_epsilon
from src.utils import set_seed

logger = logging.getLogger(__name__)

app = ClientApp()

ACCOUNTANT_STATE_KEY = "pldp_accountant_state"
SCHEDULER_STATE_KEY = "pldp_scheduler_state"

_OPTIMIZATION_METRIC_KEY_MAP: dict[str, str] = {
    "nun": "update_norm",
    "utility": "utility_loss",
}


def _make_scheduler(
    partition_id: int,
    train_dataset: object,
    config,
    _num_partitions: int,
) -> EpsilonScheduler | None:
    if not config.privacy.enabled:
        return None
    if config.bo.enabled:
        return PLDPBOScheduler(
            epsilon_min=config.bo.epsilon_min,
            epsilon_max=config.bo.epsilon_max,
            warmup_rounds=config.bo.warmup_rounds,
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
        )
        return FixedEpsilonScheduler(epsilon)
    if config.privacy.target_epsilon is not None:
        return FixedEpsilonScheduler(config.privacy.target_epsilon)
    return None


def _restore_or_create_scheduler(
    context: Context,
    partition_id: int,
    train_dataset: object,
    config,
    num_partitions: int,
) -> EpsilonScheduler | None:
    if SCHEDULER_STATE_KEY in context.state:
        state = context.state[SCHEDULER_STATE_KEY]
        stype = state.get("type")
        if stype == "fixed":
            return FixedEpsilonScheduler.from_state(state)
        elif stype == "uniform_random":
            return UniformRandomEpsilonScheduler.from_state(state)
        elif stype == "pldp_bo":
            return PLDPBOScheduler.from_state(state)
        raise ValueError(f"Unknown scheduler type: {stype}")
    return _make_scheduler(partition_id, train_dataset, config, num_partitions)


@app.train()
def train(msg: Message, context: Context) -> Message:
    config_path = str(context.run_config.get("config-path", "config/default.yaml"))
    config = load_config(config_path)

    set_seed(config.seed, deterministic=config.deterministic)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    trainloader, valloader, train_dataset = create_client_dataloader(
        config.data, partition_id, num_partitions, config.seed
    )

    accountant: RDPAccountant | None = None
    scheduler: EpsilonScheduler | None = None

    if config.privacy.enabled:
        if ACCOUNTANT_STATE_KEY in context.state:
            state = context.state[ACCOUNTANT_STATE_KEY]
            accountant = RDPAccountant.from_state(state)
        else:
            accountant = RDPAccountant(delta=config.privacy.delta)

        scheduler = _restore_or_create_scheduler(
            context, partition_id, train_dataset, config, num_partitions,
        )
        if scheduler is not None:
            logger.info(
                "Client %d scheduler: %s",
                partition_id,
                scheduler,
            )

    epsilon = _resolve_epsilon(scheduler, accountant, config)
    if epsilon < 0:
        logger.info(
            "Client %d privacy budget exhausted (epsilon=%.4f), ceasing participation",
            partition_id,
            epsilon,
        )
        epsilon = 0.0
    logger.debug("Client %d using epsilon=%.4f", partition_id, epsilon)

    total_budget: float | None = None
    if config.bo.enabled:
        total_budget = config.bo.epsilon_budget
    elif scheduler is not None:
        total_budget = epsilon

    client_model = create_model(config.model)
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

    arrays = msg.content["arrays"]
    assert isinstance(arrays, ArrayRecord)
    parameters = arrays.to_numpy_ndarrays()
    parameters_prime, num_examples, fit_metrics = client.fit(parameters, {})

    if scheduler is not None and accountant is not None:
        if not fit_metrics.get("budget_exhausted", False):
            metric_key = _OPTIMIZATION_METRIC_KEY_MAP.get(
                config.bo.optimization_metric, config.bo.optimization_metric,
            )
            metric_value = fit_metrics.get(metric_key)
            if metric_value is not None:
                scheduler.step(epsilon, float(metric_value))

    if accountant is not None and config.privacy.enabled:
        context.state[ACCOUNTANT_STATE_KEY] = accountant.get_state()

    if scheduler is not None and config.privacy.enabled:
        context.state[SCHEDULER_STATE_KEY] = scheduler.get_state()

    model_record = ArrayRecord(client_model.get_model().state_dict())
    metrics = {
        "num-examples": num_examples,
        "client-id": partition_id,
        **fit_metrics,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


def _resolve_epsilon(
    scheduler: EpsilonScheduler | None,
    accountant: RDPAccountant | None,
    config,
) -> float:
    if scheduler is not None:
        candidate = scheduler.get_epsilon()
    elif config.privacy.target_epsilon is not None:
        candidate = config.privacy.target_epsilon
    else:
        return config.privacy.noise_multiplier

    if accountant is not None and config.bo.enabled:
        budget = config.bo.epsilon_budget
        eps_min = config.bo.epsilon_min
        C = config.privacy.max_grad_norm
        delta = config.privacy.delta
        candidate = enforce_epsilon_budget(
            candidate, accountant.rdp_per_alpha, budget, eps_min, C, delta,
        )

    return candidate


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    config_path = str(context.run_config.get("config-path", "config/default.yaml"))
    config = load_config(config_path)

    set_seed(config.seed, deterministic=config.deterministic)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    trainloader, valloader, _ = create_client_dataloader(
        config.data, partition_id, num_partitions, config.seed
    )

    client_epsilon = None
    if config.privacy.enabled and config.personalization.enabled:
        if ACCOUNTANT_STATE_KEY in context.state:
            state = context.state[ACCOUNTANT_STATE_KEY]
            accountant = RDPAccountant.from_state(state)
            client_epsilon = accountant.get_epsilon()

    client_model = create_model(config.model)
    client = create_client(
        cid=partition_id,
        model=client_model,
        trainloader=trainloader,
        valloader=valloader,
        config=config,
    )

    arrays = msg.content["arrays"]
    assert isinstance(arrays, ArrayRecord)
    parameters = arrays.to_numpy_ndarrays()
    loss, num_examples, eval_metrics = client.evaluate(parameters, {})

    metrics = {
        "loss": loss,
        "num-examples": num_examples,
        "client-id": partition_id,
        **eval_metrics,
    }
    if client_epsilon is not None:
        metrics["cumulative_epsilon"] = client_epsilon
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
