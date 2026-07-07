from pathlib import Path
from typing import Optional

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from src.plotting._helpers import extract_metrics_by_round, get_run_by_id


def plot_privacy_budget(
    run_id: str,
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> matplotlib.figure.Figure:
    run = get_run_by_id(run_id)

    rounds, epsilons = extract_metrics_by_round(run, "epsilon")

    if not rounds:
        raise ValueError(f"No epsilon metrics found in run {run_id}")

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
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_client_epsilon_distribution(
    run_id: str,
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> matplotlib.figure.Figure:
    run = get_run_by_id(run_id)

    client_epsilons: dict[int, tuple[int, float]] = {}
    for key, value in run.data.metrics.items():
        if "_client_" in key and key.endswith("_epsilon") and value > 0:
            parts = key.split("_")
            try:
                round_num = int(parts[1])
                client_id = int(parts[3])
            except (ValueError, IndexError):
                continue
            prev_round, _ = client_epsilons.get(client_id, (-1, 0.0))
            if round_num > prev_round:
                client_epsilons[client_id] = (round_num, float(value))

    if not client_epsilons:
        raise ValueError(f"No per-client epsilon metrics found in run {run_id}")

    client_ids = sorted(client_epsilons.keys())
    epsilons = [client_epsilons[cid][1] for cid in client_ids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(client_ids)))
    ax1.bar(client_ids, epsilons, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Client ID")
    ax1.set_ylabel("Epsilon (ε)")
    ax1.set_title("Per-Client Epsilon Budget")
    ax1.set_xticks(client_ids)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.hist(
        epsilons,
        bins=min(10, len(epsilons)),
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xlabel("Epsilon (ε)")
    ax2.set_ylabel("Number of Clients")
    ax2.set_title("Epsilon Distribution Across Clients")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Personalized Privacy Budgets (n={len(client_ids)} clients)", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_cumulative_privacy_budget(
    run_id: str,
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> matplotlib.figure.Figure:
    run = get_run_by_id(run_id)

    client_cumulative: dict[int, list[tuple[int, float]]] = {}
    for key, value in run.data.metrics.items():
        if key.endswith("_cumulative_epsilon") and value > 0:
            parts = key.split("_")
            try:
                round_num = int(parts[1])
                client_id = int(parts[3])
            except (ValueError, IndexError):
                continue
            if client_id not in client_cumulative:
                client_cumulative[client_id] = []
            client_cumulative[client_id].append((round_num, float(value)))

    if not client_cumulative:
        raise ValueError(f"No cumulative epsilon metrics found in run {run_id}")

    fig, ax = plt.subplots(figsize=(10, 6))

    num_clients = len(client_cumulative)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_clients)))
    for i, (client_id, data) in enumerate(sorted(client_cumulative.items())):
        data.sort(key=lambda x: x[0])
        rounds_list = [d[0] for d in data]
        epsilons_list = [d[1] for d in data]
        ax.plot(
            rounds_list, epsilons_list,
            marker="o", markersize=2,
            label=f"Client {client_id}",
            color=colors[i % len(colors)],
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Epsilon (ε)")
    ax.set_title("Cumulative Privacy Budget Per Client")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
