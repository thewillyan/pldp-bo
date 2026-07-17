import warnings
from pathlib import Path

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

from src.plotting._helpers import extract_metrics_by_round, get_run_by_id


def plot_convergence(
    run_id: str,
    save_path: Path | None = None,
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

    has_loss = bool(losses)
    has_acc = bool(accuracies)
    ncols = int(has_loss) + int(has_acc)

    if ncols == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        axes = [ax]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        axes = [ax1, ax2]

    all_rounds = list(rounds) + list(acc_rounds)
    x_min, x_max = min(all_rounds), max(all_rounds)
    pad = 0.02 * (x_max - x_min) or 0.5

    ax_idx = 0
    if has_loss:
        ax = axes[ax_idx]
        ax.plot(rounds, losses, marker="o", markersize=3)
        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.set_title("Server Loss vs Round")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax_idx += 1

    if has_acc:
        ax = axes[ax_idx]
        ax.plot(acc_rounds, accuracies, marker="o", markersize=3, color="green")
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Round")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min - pad, x_max + pad)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
