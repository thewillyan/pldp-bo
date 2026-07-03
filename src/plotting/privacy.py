from pathlib import Path
from typing import Optional

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

from src.plotting._helpers import extract_metrics_by_round, get_run_by_id


def plot_privacy_budget(
    experiment_id: str,
    save_path: Optional[Path] = None,
) -> matplotlib.figure.Figure:
    run = get_run_by_id(experiment_id)

    rounds, epsilons = extract_metrics_by_round(run, "epsilon")

    if not rounds:
        raise ValueError(f"No epsilon metrics found in experiment {experiment_id}")

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
