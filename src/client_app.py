from __future__ import annotations

import logging

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.client import create_client
from src.config.loader import load_config
from src.data import create_client_dataloader
from src.models import create_model
from src.privacy.accountant import RDPAccountant
from src.privacy.personalization import assign_epsilon
from src.utils import set_seed

logger = logging.getLogger(__name__)

app = ClientApp()

ACCOUNTANT_STATE_KEY = "pldp_accountant_state"


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

    client_epsilon = None
    accountant = None

    if config.privacy.enabled and config.personalization.enabled:
        client_epsilon = assign_epsilon(
            partition_id,
            train_dataset,
            config.personalization,
            num_clients=config.data.num_clients,
        )
        logger.info(
            "Client %d assigned epsilon=%.4f (strategy=%s)",
            partition_id,
            client_epsilon,
            config.personalization.strategy,
        )

        if config.personalization.track_cumulative:
            if ACCOUNTANT_STATE_KEY in context.state:
                state = context.state[ACCOUNTANT_STATE_KEY]
                accountant = RDPAccountant.from_state(state)
            else:
                accountant = RDPAccountant(delta=config.privacy.delta)

    client_model = create_model(config.model)
    client = create_client(
        cid=partition_id,
        model=client_model,
        trainloader=trainloader,
        valloader=valloader,
        config=config,
        client_epsilon=client_epsilon,
        accountant=accountant,
        num_rounds=config.federated.num_rounds,
    )

    arrays = msg.content["arrays"]
    assert isinstance(arrays, ArrayRecord)
    parameters = arrays.to_numpy_ndarrays()
    parameters_prime, num_examples, fit_metrics = client.fit(parameters, {})

    if (
        config.privacy.enabled
        and config.personalization.enabled
        and config.personalization.track_cumulative
        and accountant is not None
    ):
        context.state[ACCOUNTANT_STATE_KEY] = accountant.get_state()

    model_record = ArrayRecord(client_model.get_model().state_dict())
    metrics = {
        "num-examples": num_examples,
        "client-id": partition_id,
        **fit_metrics,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


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
