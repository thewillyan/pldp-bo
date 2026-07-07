from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.plotting._helpers import (
    extract_per_client_metric,
    extract_round_stats,
    get_run_by_id,
)


def plot_epsilon_schedules(
    run_id: str,
    client_ids: Optional[list[int]] = None,
    show_mean_std: bool = True,
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    run = get_run_by_id(run_id)

    if client_ids is None:
        seen: set[int] = set()
        for key in run.data.metrics:
            parts = key.split("_")
            if (
                len(parts) >= 5
                and parts[0] == "round"
                and parts[2] == "client"
                and parts[4] == "epsilon"
            ):
                try:
                    seen.add(int(parts[3]))
                except ValueError:
                    continue
        client_ids = sorted(seen)

    if not client_ids:
        msg = "No per-client epsilon metrics found. Run the experiment first."
        raise ValueError(msg)

    fig, ax = plt.subplots(figsize=(10, 6))

    client_data: dict[int, tuple[list[int], list[float]]] = {}

    for cid in client_ids:
        rounds, epsilons = extract_per_client_metric(run, cid, "epsilon")
        if rounds:
            client_data[cid] = (rounds, epsilons)

    if not client_data:
        raise ValueError(f"No epsilon data for clients {client_ids}")

    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(client_data)))
    for i, (cid, (rds, eps)) in enumerate(sorted(client_data.items())):
        ax.plot(
            rds, eps,
            marker=".", markersize=4, linewidth=1.0, alpha=0.7,
            color=palette[i], label=f"Client {cid}",
        )

    if show_mean_std and len(client_data) > 1:
        sorted_rounds, stats = extract_round_stats(run, "epsilon")
        if sorted_rounds:
            mean_vals = stats.get("mean", [])
            std_vals = stats.get("std", [])
            if mean_vals and std_vals:
                ax.plot(
                    sorted_rounds, mean_vals,
                    color="black", linewidth=2.5, label="Mean", linestyle="--",
                )
                upper = [m + s for m, s in zip(mean_vals, std_vals, strict=True)]
                lower = [m - s for m, s in zip(mean_vals, std_vals, strict=True)]
                ax.fill_between(sorted_rounds, lower, upper, alpha=0.15, color="black", label="±1σ")

    ax.set_xlabel("Round")
    ax.set_ylabel("Epsilon (ε)")
    ax.set_title("Per-Client Epsilon Schedules")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_metric_vs_epsilon(
    run_id: str,
    client_id: int = 0,
    metric: str = "utility_loss",
    warmup_rounds: Optional[int] = None,
    save_path: Optional[Path] = None,
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
        ax.scatter(eps_arr, met_arr, c=rds_arr, cmap="viridis", marker="o", s=50, alpha=0.7)

    ax.set_xlabel("Epsilon (ε)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Client {client_id}: {metric.replace('_', ' ').title()} vs ε")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
