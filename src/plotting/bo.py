from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plotting._helpers import (
    extract_per_client_metric,
    get_run_by_id,
)


def plot_metric_vs_epsilon(
    run_id: str,
    client_id: int = 0,
    metric: str = "utility_loss",
    warmup_rounds: int | None = None,
    save_path: Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    run = get_run_by_id(run_id)

    rounds, epsilons = extract_per_client_metric(run, client_id, "epsilon")
    _, metric_vals = extract_per_client_metric(run, client_id, metric)

    if not rounds:
        msg = f"No data for client {client_id} metric '{metric}'"
        raise ValueError(msg)

    if not metric_vals:
        msg = f"No data for client {client_id} metric '{metric}'"
        raise ValueError(msg)

    if len(epsilons) != len(metric_vals):
        msg = (
            f"Length mismatch for client {client_id}: "
            f"{len(epsilons)} epsilon values vs {len(metric_vals)} {metric} values"
        )
        raise ValueError(msg)

    fig, ax = plt.subplots(figsize=(8, 6))

    rds_arr = np.array(rounds)
    eps_arr = np.array(epsilons)
    met_arr = np.array(metric_vals)

    if warmup_rounds is not None:
        warmup_idx = (
            rds_arr <= warmup_rounds
            if warmup_rounds > 0
            else np.zeros_like(rds_arr, dtype=bool)
        )

        if warmup_idx.any():
            ax.scatter(
                eps_arr[warmup_idx], met_arr[warmup_idx],
                c=rds_arr[warmup_idx], cmap="Blues", marker="o", s=50, alpha=0.7,
                label="Warm-up",
            )
        if (~warmup_idx).any():
            ax.scatter(
                eps_arr[~warmup_idx], met_arr[~warmup_idx],
                c=rds_arr[~warmup_idx], cmap="Reds", marker="s", s=50, alpha=0.7,
                label="BO",
            )

        ax.legend(framealpha=0.9)
    else:
        sc = ax.scatter(eps_arr, met_arr, c=rds_arr, cmap="viridis", marker="o", s=50, alpha=0.7)
        fig.colorbar(sc, ax=ax, label="Round")

    ax.set_xlabel("Epsilon (ε)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Client {client_id}: {metric.replace('_', ' ').title()} vs ε")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
