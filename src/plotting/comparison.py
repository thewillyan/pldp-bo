import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy
import seaborn as sns

from src.plotting._helpers import (
    extract_metrics_by_round,
    extract_round_stats,
    get_run_by_id,
    get_run_name,
)

LINE_STYLES = ["-", "--", "-.", ":"]
PALETTE = "husl"


def _resolve_labels(
    run_ids: Sequence[str],
    labels: Sequence[str] | None,
) -> list[str]:
    if labels is not None:
        if len(labels) != len(run_ids):
            msg = f"Expected {len(run_ids)} labels, got {len(labels)}"
            raise ValueError(msg)
        return list(labels)
    return [get_run_name(get_run_by_id(rid)) for rid in run_ids]


def _setup_figure(
    n_plots: int,
    n_runs: int,
) -> tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]:
    if n_plots == 1:
        width = 8
    elif n_plots == 2:
        width = 14
    else:
        width = 7 * n_plots
    height = max(5, 1.5 * n_runs)
    fig, axes = plt.subplots(1, n_plots, figsize=(width, height))
    if n_plots == 1:
        axes = [axes]
    return fig, list(axes)


def plot_comparison_convergence(
    run_ids: Sequence[str],
    labels: Sequence[str] | None = None,
    save_path: Path | None = None,
    dpi: int = 150,
) -> matplotlib.figure.Figure:
    resolved_labels = _resolve_labels(run_ids, labels)
    palette = sns.color_palette(PALETTE, n_colors=len(run_ids))

    fig, (ax_loss, ax_acc) = _setup_figure(2, len(run_ids))

    run_data: list[tuple[str, list[int], list[float], list[int], list[float]]] = []
    for run_id, label in zip(run_ids, resolved_labels, strict=True):
        run = get_run_by_id(run_id)
        loss_rounds, losses = extract_metrics_by_round(run, "server_loss")
        acc_rounds, accuracies = extract_metrics_by_round(run, "accuracy")
        run_data.append((label, loss_rounds, losses, acc_rounds, accuracies))

    has_loss = any(rd[1] for rd in run_data)
    has_acc = any(rd[3] for rd in run_data)

    if not has_loss and not has_acc:
        msg = "No convergence metrics (server_loss, accuracy) found in any run."
        raise ValueError(msg)
    if not has_loss:
        warnings.warn(
            "No server_loss data in any run; plotting accuracy only",
            stacklevel=2,
        )
    if not has_acc:
        warnings.warn(
            "No accuracy data in any run; plotting loss only",
            stacklevel=2,
        )

    for i, (label, loss_rounds, losses, acc_rounds, accuracies) in enumerate(run_data):
        color = palette[i]
        ls = LINE_STYLES[i % len(LINE_STYLES)]

        if loss_rounds:
            ax_loss.plot(
                loss_rounds, losses, color=color, linestyle=ls, linewidth=2, label=label,
            )

        if acc_rounds:
            ax_acc.plot(
                acc_rounds,
                accuracies,
                color=color,
                linestyle=ls,
                linewidth=2,
                label=label,
            )

    all_rounds: list[int] = []
    for _, loss_rounds, _, acc_rounds, _ in run_data:
        all_rounds.extend(loss_rounds)
        all_rounds.extend(acc_rounds)
    if all_rounds:
        x_min, x_max = min(all_rounds), max(all_rounds)
        pad = 0.02 * (x_max - x_min) or 0.5
        ax_loss.set_xlim(x_min - pad, x_max + pad)
        ax_acc.set_xlim(x_min - pad, x_max + pad)

    ax_loss.set_xlabel("Round")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Server Loss vs Round")
    ax_loss.legend(frameon=True, framealpha=0.9, edgecolor="gray")

    ax_acc.set_xlabel("Round")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy vs Round")
    ax_acc.legend(frameon=True, framealpha=0.9, edgecolor="gray")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_mean_convergence(
    run_ids: Sequence[str],
    labels: Sequence[str] | None = None,  # noqa: ARG001
    save_path: Path | None = None,
    dpi: int = 150,
) -> matplotlib.figure.Figure:

    all_loss: list[list[float]] = []
    all_acc: list[list[float]] = []
    common_rounds_loss: list[int] = []
    common_rounds_acc: list[int] = []

    for run_id in run_ids:
        run = get_run_by_id(run_id)
        loss_rounds, losses = extract_metrics_by_round(run, "server_loss")
        acc_rounds, accuracies = extract_metrics_by_round(run, "accuracy")
        if loss_rounds and not common_rounds_loss:
            common_rounds_loss = loss_rounds
        if acc_rounds and not common_rounds_acc:
            common_rounds_acc = acc_rounds

        if losses and len(losses) == len(common_rounds_loss):
            all_loss.append(losses)
        if accuracies and len(accuracies) == len(common_rounds_acc):
            all_acc.append(accuracies)

    has_loss = len(all_loss) > 0
    has_acc = len(all_acc) > 0

    if not has_loss and not has_acc:
        msg = "No convergence metrics (server_loss, accuracy) found in any run."
        raise ValueError(msg)
    if not has_loss:
        warnings.warn("No server_loss data; plotting accuracy only", stacklevel=2)
    if not has_acc:
        warnings.warn("No accuracy data; plotting loss only", stacklevel=2)

    ncols = int(has_loss) + int(has_acc)
    if ncols == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        axes = [ax]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        axes = [ax1, ax2]

    x_min, x_max = 0, 0
    if has_loss and common_rounds_loss:
        x_min, x_max = min(common_rounds_loss), max(common_rounds_loss)
    if has_acc and common_rounds_acc:
        acc_min, acc_max = min(common_rounds_acc), max(common_rounds_acc)
        x_min = min(x_min, acc_min) if has_loss else acc_min
        x_max = max(x_max, acc_max) if has_loss else acc_max
    pad = 0.02 * (x_max - x_min) or 0.5

    ax_idx = 0
    if has_loss:
        ax = axes[ax_idx]
        losses_arr = numpy.array(all_loss)
        mean = numpy.mean(losses_arr, axis=0)
        n_contrib = len(all_loss)
        ax.plot(common_rounds_loss, mean, color="steelblue", linewidth=2,
                label=f"Mean (N={n_contrib})")
        if n_contrib > 1:
            std = numpy.std(losses_arr, axis=0, ddof=1)
            ax.fill_between(common_rounds_loss, mean - std, mean + std,
                            alpha=0.3, color="steelblue")
        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.set_title("Server Loss vs Round (Mean ± Std)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.legend(frameon=True, framealpha=0.9, edgecolor="gray")
        ax_idx += 1

    if has_acc:
        ax = axes[ax_idx]
        acc_arr = numpy.array(all_acc)
        mean = numpy.mean(acc_arr, axis=0)
        n_contrib = len(all_acc)
        ax.plot(common_rounds_acc, mean, color="green", linewidth=2,
                label=f"Mean (N={n_contrib})")
        if n_contrib > 1:
            std = numpy.std(acc_arr, axis=0, ddof=1)
            ax.fill_between(common_rounds_acc, mean - std, mean + std,
                            alpha=0.3, color="green")
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Round (Mean ± Std)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.legend(frameon=True, framealpha=0.9, edgecolor="gray")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_comparison_privacy(
    run_ids: Sequence[str],
    labels: Sequence[str] | None = None,
    save_path: Path | None = None,
    dpi: int = 150,
    show_std: bool = True,
) -> matplotlib.figure.Figure:
    resolved_labels = _resolve_labels(run_ids, labels)
    palette = sns.color_palette(PALETTE, n_colors=len(run_ids))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    has_per_round = False
    has_cumulative = False
    lines: list = []
    legend_labels: list[str] = []

    for i, (run_id, label) in enumerate(
        zip(run_ids, resolved_labels, strict=True),
    ):
        run = get_run_by_id(run_id)
        color = palette[i]
        ls = LINE_STYLES[i % len(LINE_STYLES)]

        aggs = ("mean", "std") if show_std else ("mean",)

        rounds_per, eps_stats = extract_round_stats(run, "epsilon", aggs=aggs)
        epsilons = eps_stats.get("mean", [])
        if epsilons:
            (line_per,) = ax1.plot(
                rounds_per, epsilons, color=color, linestyle=ls,
                linewidth=2, label=f"{label} (ε/round)",
            )
            lines.append(line_per)
            legend_labels.append(f"{label} (ε/round)")
            has_per_round = True

            if show_std:
                stds = eps_stats.get("std", [])
                if stds and len(stds) == len(epsilons):
                    upper = [m + s for m, s in zip(epsilons, stds, strict=True)]
                    lower = [m - s for m, s in zip(epsilons, stds, strict=True)]
                    ax1.fill_between(rounds_per, lower, upper, alpha=0.1, color=color)

        rounds_cum, cum_stats = extract_round_stats(run, "cumulative_epsilon", aggs=aggs)
        cum_epsilons = cum_stats.get("mean", [])
        if cum_epsilons:
            (line_cum,) = ax2.plot(
                rounds_cum, cum_epsilons, color=color,
                linestyle="--", linewidth=2, label=f"{label} (cumulative ε)",
            )
            lines.append(line_cum)
            legend_labels.append(f"{label} (cumulative ε)")
            has_cumulative = True

    if not has_per_round and not has_cumulative:
        msg = "No epsilon or cumulative_epsilon metrics found in any of the specified runs."
        raise ValueError(msg)

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Per-Round Epsilon (ε)")
    ax2.set_ylabel("Cumulative Epsilon (ε)")

    title = "Privacy Budget vs Round"
    if has_per_round and has_cumulative:
        title += " — Per-Round (left) & Cumulative (right)"
    elif has_per_round:
        title += " — Per-Round Epsilon"
    else:
        title += " — Cumulative Epsilon"
    ax1.set_title(title)

    if lines:
        ax1.legend(
            lines, legend_labels,
            frameon=True, framealpha=0.9, edgecolor="gray",
        )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
