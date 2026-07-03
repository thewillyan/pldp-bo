from __future__ import annotations

import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from src.config.loader import load_config
from src.data import create_client_dataloaders
from src.logging.tracker import ExperimentTracker
from src.models import create_model
from src.utils import set_seed


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    config_path = str(context.run_config.get("config-path", "config/default.yaml"))
    config = load_config(config_path)

    set_seed(config.seed, deterministic=config.deterministic)

    _, valloader, _ = create_client_dataloaders(config.data, config.seed)

    tracker = ExperimentTracker(config)
    tracker.start_run()

    global_model = create_model(config.model)
    arrays = ArrayRecord(global_model.get_model().state_dict())

    if config.federated.strategy == "fedprox":
        strategy = FedProx(
            fraction_train=config.federated.fraction_fit,
            fraction_evaluate=config.federated.fraction_evaluate,
            min_train_nodes=config.federated.min_fit_clients,
            min_evaluate_nodes=config.federated.min_evaluate_clients,
            min_available_nodes=config.federated.min_available_nodes,
            proximal_mu=config.federated.proximal_mu,
        )
    else:
        strategy = FedAvg(
            fraction_train=config.federated.fraction_fit,
            fraction_evaluate=config.federated.fraction_evaluate,
            min_train_nodes=config.federated.min_fit_clients,
            min_evaluate_nodes=config.federated.min_evaluate_clients,
            min_available_nodes=config.federated.min_available_nodes,
        )

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = create_model(config.model)
        model.get_model().load_state_dict(arrays.to_torch_state_dict())
        net = model.get_model()
        net.eval()

        criterion = torch.nn.CrossEntropyLoss()
        loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in valloader:
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = loss / len(valloader)

        metrics: dict[str, float] = {
            "server_loss": avg_loss,
            "accuracy": accuracy,
        }
        tracker.log_round_metrics(server_round, metrics)

        return MetricRecord({"accuracy": accuracy, "loss": avg_loss})

    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=config.federated.num_rounds,
        evaluate_fn=global_evaluate,
    )

    tracker.end_run()
