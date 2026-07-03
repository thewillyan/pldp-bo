from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow


def plot_privacy_budget(
    experiment_id: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(experiment_id)

    rounds = []
    epsilons = []

    for key, value in run.data.metrics.items():
        if key.startswith("round_") and key.endswith("_epsilon"):
            parts = key.split("_")
            if len(parts) >= 3:
                round_num = int(parts[1])
                rounds.append(round_num)
                epsilons.append(float(value))

    if not rounds:
        raise ValueError(f"No epsilon metrics found in experiment {experiment_id}")

    sorted_data = sorted(zip(rounds, epsilons))
    rounds, epsilons = zip(*sorted_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(rounds, epsilons, marker="o", markersize=3, color="red")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Epsilon (ε)")
    ax1.set_title("Privacy Budget (ε) vs Round")
    ax1.grid(True, alpha=0.3)

    if len(epsilons) > 1:
        ax2.plot(epsilons, range(len(epsilons)), marker="o", markersize=3, color="red")
        ax2.set_xlabel("Epsilon (ε)")
        ax2.set_ylabel("Round")
        ax2.set_title("Round vs Privacy Budget")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
