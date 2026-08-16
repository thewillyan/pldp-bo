from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from src.plotting._helpers import (
    _get_metric_history,
    _is_rdp_native,
    extract_round_stats,
    get_run_by_id,
)


def _discover_client_ids(run) -> set[int]:
    ids: set[int] = set()
    for key in run.data.metrics:
        if "_client_" not in key:
            continue
        parts = key.split("_")
        try:
            idx = parts.index("client")
            if idx + 1 < len(parts):
                ids.add(int(parts[idx + 1]))
        except ValueError, IndexError:
            continue
    return ids


def _get_client_latest(run, metric_suffix: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for key, value in run.data.metrics.items():
        if not key.endswith(metric_suffix):
            continue
        parts = key.split("_")
        try:
            idx = parts.index("client")
            cid = int(parts[idx + 1])
            result[cid] = float(value)
        except ValueError, IndexError:
            continue
    return result


def _get_client_trace(run_id: str, client_id: int, metric: str) -> list[tuple[int, float]]:
    history = _get_metric_history(run_id, f"client_{client_id}_{metric}")
    if history:
        return sorted(set((step, val) for step, val in history), key=lambda x: x[0])
    return []


def _detect_format(run) -> str:
    for key in run.data.metrics:
        if key.startswith("round_"):
            return "legacy"
    return "new"


def plot_client_epsilon_distribution(
    run_id: str,
    save_path: Path | None = None,
    dpi: int = 150,
) -> matplotlib.figure.Figure:
    run = get_run_by_id(run_id)
    fmt = _detect_format(run)

    if fmt == "legacy":
        return _plot_client_epsilon_legacy(run, run_id, save_path, dpi)

    rdp = _is_rdp_native(run)
    remaining_suffix = "_remaining_rdp_budget" if rdp else "_remaining_budget"
    cumulative_suffix = "_cumulative_rdp" if rdp else "_cumulative_epsilon"
    trace_metric = "rdp_cost" if rdp else "epsilon"
    stat_metric = "rdp_cost" if rdp else "epsilon"
    x_label = "RDP(α)" if rdp else "Epsilon (ε)"

    remaining = _get_client_latest(run, remaining_suffix)
    used_data = _get_client_latest(run, cumulative_suffix)

    client_trace: dict[int, list[tuple[int, float]]] = {}
    for cid in _discover_client_ids(run):
        trace = _get_client_trace(run_id, cid, trace_metric)
        if trace:
            client_trace[cid] = trace

    all_ids = sorted(remaining.keys() | used_data.keys())

    if remaining and used_data:
        allocated_vals = [remaining.get(cid, 0.0) + used_data.get(cid, 0.0) for cid in all_ids]
        used_vals = [used_data.get(cid, 0.0) for cid in all_ids]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        x = np.arange(len(all_ids))
        width = 0.35

        ax1.bar(
            x - width / 2,
            allocated_vals,
            width,
            label="Initial Budget",
            color="#4A90D9",
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.bar(
            x + width / 2,
            used_vals,
            width,
            label="Cumulative Used",
            color="#1A5276",
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.set_xlabel("Client ID")
        ax1.set_ylabel(x_label)
        ax1.set_title("Privacy Budget: Initial Allocation vs Cumulative Spend")
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_ids)
        ax1.legend(frameon=True, framealpha=0.9, edgecolor="gray")
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.text(
            0.5,
            -0.22,
            "Initial Budget = Remaining Budget + Cumulative Used",
            transform=ax1.transAxes,
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

        total_rounds_str = run.data.params.get("federated.num_rounds", "0")
        total_rounds = int(total_rounds_str) if total_rounds_str else 0
        trace_colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(all_ids))))
        for i, cid in enumerate(sorted(all_ids)):
            data = sorted(client_trace.get(cid, []))
            if not data:
                continue
            rds = [d[0] for d in data]
            eps = [d[1] for d in data]
            color = trace_colors[i % len(trace_colors)]
            ax2.plot(
                rds,
                eps,
                marker="o",
                markersize=4,
                color=color,
                label=f"Client {cid}",
                linewidth=1.5,
            )
            last_r, last_eps = data[-1]
            exhausted = total_rounds > 0 and (last_r < total_rounds - 1 or len(data) < total_rounds)
            marker_color = "#E74C3C" if exhausted else color
            ax2.plot(last_r, last_eps, marker="x", markersize=8, color=marker_color, mew=2)

        sorted_rounds, stats = extract_round_stats(run, stat_metric, aggs=("mean", "std"))
        if sorted_rounds:
            mean_vals = stats.get("mean", [])
            std_vals = stats.get("std", [])
            if mean_vals and std_vals and len(mean_vals) == len(std_vals):
                ax2.plot(
                    sorted_rounds,
                    mean_vals,
                    color="black",
                    linewidth=2.5,
                    linestyle="--",
                    label="Mean ± σ",
                    zorder=len(all_ids) + 5,
                )
                upper = [m + s for m, s in zip(mean_vals, std_vals, strict=False)]
                lower = [m - s for m, s in zip(mean_vals, std_vals, strict=False)]
                ax2.fill_between(
                    sorted_rounds, lower, upper, alpha=0.15, color="black", zorder=len(all_ids) + 4
                )

        ax2.set_xlabel("Round")
        ax2.set_ylabel(x_label)
        ax2.set_title("Privacy Cost Expended per Round")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        ax2.grid(alpha=0.3)

        plt.suptitle(
            f"Personalized Privacy Budgets (n={len(all_ids)} clients)",
            fontsize=13,
        )
        plt.subplots_adjust(bottom=0.12)
        plt.tight_layout(rect=(0, 0.06, 1, 1))

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig

    if remaining and not used_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        ids_sorted = sorted(remaining.keys())
        vals = [remaining[cid] for cid in ids_sorted]
        ax.bar(ids_sorted, vals, color="#4A90D9", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Client ID")
        ax.set_ylabel(f"Remaining {x_label}")
        ax.set_title("Privacy Budget Remaining Per Client")
        ax.set_xticks(ids_sorted)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    if not remaining and used_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        ids_sorted = sorted(used_data.keys())
        vals = [used_data[cid] for cid in ids_sorted]
        ax.bar(ids_sorted, vals, color="#1A5276", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Client ID")
        ax.set_ylabel(f"Cumulative {x_label}")
        ax.set_title("Privacy Budget Used Per Client")
        ax.set_xticks(ids_sorted)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    raise ValueError(f"No per-client privacy metrics found in run {run_id}")


def _plot_client_epsilon_legacy(
    run,
    run_id: str,
    save_path: Path | None,
    dpi: int,
) -> matplotlib.figure.Figure:
    remaining: dict[int, tuple[int, float]] = {}
    used_data: dict[int, tuple[int, float]] = {}
    per_round: dict[int, tuple[int, float]] = {}
    client_trace: dict[int, list[tuple[int, float]]] = {}

    for key, value in run.data.metrics.items():
        if "_client_" not in key:
            continue
        parts = key.split("_")
        if len(parts) < 5 or parts[0] != "round" or parts[2] != "client":
            continue
        try:
            round_num = int(parts[1])
            client_id = int(parts[3])
        except ValueError, IndexError:
            continue

        if key.endswith("_remaining_budget"):
            prev_round, _ = remaining.get(client_id, (-1, float("inf")))
            if round_num > prev_round:
                remaining[client_id] = (round_num, float(value))
        elif key.endswith("_cumulative_epsilon"):
            prev_round, _ = used_data.get(client_id, (-1, 0.0))
            if round_num > prev_round:
                used_data[client_id] = (round_num, float(value))
        elif (
            key.endswith("_epsilon")
            and not key.endswith("_client_epsilon")
            and not key.endswith("_cumulative_epsilon")
        ):
            prev_round, _ = per_round.get(client_id, (-1, 0.0))
            if round_num > prev_round:
                per_round[client_id] = (round_num, float(value))
            client_trace.setdefault(client_id, []).append((round_num, float(value)))

    has_remaining = bool(remaining)
    has_used = bool(used_data)

    if has_remaining and has_used:
        all_ids = sorted(set(remaining.keys()) | set(used_data.keys()))
        allocated_vals = [
            remaining.get(cid, (0, 0.0))[1] + used_data.get(cid, (0, 0.0))[1] for cid in all_ids
        ]
        used_vals = [used_data.get(cid, (0, 0.0))[1] for cid in all_ids]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        x = np.arange(len(all_ids))
        width = 0.35

        ax1.bar(
            x - width / 2,
            allocated_vals,
            width,
            label="Initial Budget",
            color="#4A90D9",
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.bar(
            x + width / 2,
            used_vals,
            width,
            label="Cumulative Used",
            color="#1A5276",
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.set_xlabel("Client ID")
        ax1.set_ylabel("Epsilon (ε)")
        ax1.set_title("Privacy Budget: Initial Allocation vs Cumulative Spend")
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_ids)
        ax1.legend(frameon=True, framealpha=0.9, edgecolor="gray")
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.text(
            0.5,
            -0.22,
            "Initial Budget = Remaining Budget + Cumulative Used",
            transform=ax1.transAxes,
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

        total_rounds_str = run.data.params.get("federated.num_rounds", "0")
        total_rounds = int(total_rounds_str) if total_rounds_str else 0
        trace_colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(all_ids))))
        for i, cid in enumerate(sorted(all_ids)):
            data = sorted(client_trace.get(cid, []))
            if not data:
                continue
            rds = [d[0] for d in data]
            eps = [d[1] for d in data]
            color = trace_colors[i % len(trace_colors)]
            ax2.plot(
                rds,
                eps,
                marker="o",
                markersize=4,
                color=color,
                label=f"Client {cid}",
                linewidth=1.5,
            )
            last_r, last_eps = data[-1]
            exhausted = total_rounds > 0 and (last_r < total_rounds - 1 or len(data) < total_rounds)
            marker_color = "#E74C3C" if exhausted else color
            ax2.plot(last_r, last_eps, marker="x", markersize=8, color=marker_color, mew=2)

        sorted_rounds, stats = extract_round_stats(run, "epsilon", aggs=("mean", "std"))
        if sorted_rounds:
            mean_vals = stats.get("mean", [])
            std_vals = stats.get("std", [])
            if mean_vals and std_vals and len(mean_vals) == len(std_vals):
                ax2.plot(
                    sorted_rounds,
                    mean_vals,
                    color="black",
                    linewidth=2.5,
                    linestyle="--",
                    label="Mean ± σ",
                    zorder=len(all_ids) + 5,
                )
                upper = [m + s for m, s in zip(mean_vals, std_vals, strict=False)]
                lower = [m - s for m, s in zip(mean_vals, std_vals, strict=False)]
                ax2.fill_between(
                    sorted_rounds, lower, upper, alpha=0.15, color="black", zorder=len(all_ids) + 4
                )

        ax2.set_xlabel("Round")
        ax2.set_ylabel("Epsilon (ε)")
        ax2.set_title("Epsilon Expended per Round")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        ax2.grid(alpha=0.3)

        plt.suptitle(
            f"Personalized Privacy Budgets (n={len(all_ids)} clients)",
            fontsize=13,
        )
        plt.subplots_adjust(bottom=0.12)
        plt.tight_layout(rect=(0, 0.06, 1, 1))

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig

    if has_remaining and not has_used:
        fig, ax = plt.subplots(figsize=(10, 5))
        ids_sorted = sorted(remaining.keys())
        vals = [remaining[cid][1] for cid in ids_sorted]
        ax.bar(ids_sorted, vals, color="#4A90D9", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Remaining Epsilon (ε)")
        ax.set_title("Privacy Budget Remaining Per Client")
        ax.set_xticks(ids_sorted)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    if not has_remaining and has_used:
        fig, ax = plt.subplots(figsize=(10, 5))
        ids_sorted = sorted(used_data.keys())
        vals = [used_data[cid][1] for cid in ids_sorted]
        ax.bar(ids_sorted, vals, color="#1A5276", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Cumulative Epsilon (ε)")
        ax.set_title("Privacy Budget Used Per Client")
        ax.set_xticks(ids_sorted)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    if not per_round:
        raise ValueError(f"No per-client epsilon metrics found in run {run_id}")

    client_ids = sorted(per_round.keys())
    epsilons = [per_round[cid][1] for cid in client_ids]

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

    plt.suptitle(f"Privacy Budgets (n={len(client_ids)} clients) — legacy run", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_cumulative_privacy_budget(
    run_id: str,
    save_path: Path | None = None,
    dpi: int = 150,
) -> matplotlib.figure.Figure:
    run = get_run_by_id(run_id)
    fmt = _detect_format(run)

    if fmt == "legacy":
        return _plot_cumulative_legacy(run, run_id, save_path, dpi)

    rdp = _is_rdp_native(run)
    cumulative_metric = "cumulative_rdp" if rdp else "cumulative_epsilon"
    y_label = "Cumulative RDP(α)" if rdp else "Cumulative Epsilon (ε)"

    client_cumulative: dict[int, list[tuple[int, float]]] = {}

    for cid in _discover_client_ids(run):
        history = _get_metric_history(run_id, f"client_{cid}_{cumulative_metric}")
        if history:
            client_cumulative[cid] = sorted(set(history), key=lambda x: x[0])

    if not client_cumulative:
        raise ValueError(f"No cumulative privacy metrics found in run {run_id}")

    fig, ax = plt.subplots(figsize=(10, 6))

    num_clients = len(client_cumulative)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_clients)))
    for i, (client_id, data) in enumerate(sorted(client_cumulative.items())):
        data.sort(key=lambda x: x[0])
        rounds_list = [d[0] for d in data]
        epsilons_list = [d[1] for d in data]
        ax.plot(
            rounds_list,
            epsilons_list,
            marker="o",
            markersize=2,
            label=f"Client {client_id}",
            color=colors[i % len(colors)],
        )

    ax.set_xlabel("Round")
    ax.set_ylabel(y_label)
    ax.set_title("Cumulative Privacy Budget Per Client")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def _plot_cumulative_legacy(
    run,
    run_id: str,
    save_path: Path | None,
    dpi: int,
) -> matplotlib.figure.Figure:
    client_cumulative: dict[int, list[tuple[int, float]]] = {}
    for key, value in run.data.metrics.items():
        if key.endswith("_cumulative_epsilon") and value > 0:
            parts = key.split("_")
            try:
                round_num = int(parts[1])
                client_id = int(parts[3])
            except ValueError, IndexError:
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
            rounds_list,
            epsilons_list,
            marker="o",
            markersize=2,
            label=f"Client {client_id}",
            color=colors[i % len(colors)],
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Epsilon (ε)")
    ax.set_title("Cumulative Privacy Budget Per Client")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
