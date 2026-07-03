from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import pandas as pd


def plot_convergence(
    experiment_id: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(experiment_id)

    rounds = []
    losses = []
    accuracies = []

    for key, value in run.data.metrics.items():
        if key.startswith("round_") and "_" in key:
            parts = key.split("_")
            if len(parts) >= 3:
                round_num = int(parts[1])
                metric_name = "_".join(parts[2:])
                if metric_name == "server_loss":
                    rounds.append(round_num)
                    losses.append(float(value))
                elif metric_name == "accuracy":
                    if round_num not in [r for r, _ in zip(rounds, losses)]:
                        rounds.append(round_num)
                    accuracies.append(float(value))

    if not rounds:
        raise ValueError(f"No round metrics found in experiment {experiment_id}")

    rounds = sorted(set(rounds))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if losses:
        ax1.plot(range(len(losses)), losses, marker="o", markersize=3)
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Loss")
        ax1.set_title("Server Loss vs Round")
        ax1.grid(True, alpha=0.3)

    if accuracies:
        ax2.plot(range(len(accuracies)), accuracies, marker="o", markersize=3, color="green")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy vs Round")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
