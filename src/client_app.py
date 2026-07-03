from __future__ import annotations

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.client import create_client
from src.config.loader import load_config
from src.data import create_client_dataloader
from src.models import create_model

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    config_path = str(context.run_config.get("config-path", "config/default.yaml"))
    config = load_config(config_path)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    trainloader, valloader, _ = create_client_dataloader(
        config.data, partition_id, num_partitions, config.seed
    )

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
    parameters_prime, num_examples, fit_metrics = client.fit(parameters, {})

    model_record = ArrayRecord(client_model.get_model().state_dict())
    metrics = {"num-examples": num_examples, **fit_metrics}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    config_path = str(context.run_config.get("config-path", "config/default.yaml"))
    config = load_config(config_path)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    trainloader, valloader, _ = create_client_dataloader(
        config.data, partition_id, num_partitions, config.seed
    )

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

    metrics = {"loss": loss, "num-examples": num_examples, **eval_metrics}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
