import warnings
from pathlib import Path
from typing import Optional

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

from src.plotting._helpers import extract_metrics_by_round, get_run_by_id


def plot_convergence(
    run_id: str,
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> matplotlib.figure.Figure:
    run = get_run_by_id(run_id)

    rounds, losses = extract_metrics_by_round(run, "server_loss")
    acc_rounds, accuracies = extract_metrics_by_round(run, "accuracy")

    if not rounds and not acc_rounds:
        raise ValueError(f"No round metrics found in run {run_id}")
    if not rounds and acc_rounds:
        warnings.warn(
            f"No server_loss data for {run_id}; plotting accuracy only",
            stacklevel=2,
        )
    if rounds and not acc_rounds:
        warnings.warn(
            f"No accuracy data for {run_id}; plotting loss only",
            stacklevel=2,
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if losses:
        ax1.plot(rounds, losses, marker="o", markersize=3)
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Loss")
        ax1.set_title("Server Loss vs Round")
        ax1.grid(True, alpha=0.3)

    if accuracies:
        ax2.plot(acc_rounds, accuracies, marker="o", markersize=3, color="green")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy vs Round")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
