import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy
import seaborn as sns

from src.plotting._helpers import (
    _is_rdp_native,
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
    height = max(5, 3 * n_plots)
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

    n_plots = int(has_loss) + int(has_acc)
    fig, axes = _setup_figure(n_plots, len(run_ids))
    ax_idx = 0

    if has_loss:
        ax = axes[ax_idx]
        for i, (label, loss_rounds, loss_vals, _, _) in enumerate(run_data):
            if not loss_rounds:
                continue
            color = palette[i]
            ls = LINE_STYLES[i % len(LINE_STYLES)]
            ax.plot(
                loss_rounds, loss_vals, color=color, linestyle=ls, linewidth=2, label=label,
            )
        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.set_title("Server Loss vs Round")
        ax.legend(frameon=True, framealpha=0.9, edgecolor="gray")
        ax_idx += 1

    if has_acc:
        ax = axes[ax_idx]
        for i, (label, _, _, acc_rounds, accuracies) in enumerate(run_data):
            if not acc_rounds:
                continue
            color = palette[i]
            ls = LINE_STYLES[i % len(LINE_STYLES)]
            ax.plot(
                acc_rounds, accuracies, color=color, linestyle=ls, linewidth=2, label=label,
            )
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Round")
        ax.legend(frameon=True, framealpha=0.9, edgecolor="gray")

    all_rounds: list[int] = []
    for _, loss_rounds, _, acc_rounds, _ in run_data:
        all_rounds.extend(loss_rounds)
        all_rounds.extend(acc_rounds)
    if all_rounds:
        x_min, x_max = min(all_rounds), max(all_rounds)
        pad = 0.02 * (x_max - x_min) or 0.5
        for ax in axes:
            ax.set_xlim(x_min - pad, x_max + pad)

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

    has_per_round = False
    has_cumulative = False
    run_rdp_flags: dict[str, bool] = {}
    for run_id in run_ids:
        run = get_run_by_id(run_id)
        rdp = _is_rdp_native(run)
        run_rdp_flags[run_id] = rdp
        per_round_metric = "rdp_cost" if rdp else "epsilon"
        cumulative_metric = "cumulative_rdp" if rdp else "cumulative_epsilon"
        _, eps_stats = extract_round_stats(run, per_round_metric, aggs=("mean",))
        if eps_stats.get("mean"):
            has_per_round = True
        _, cum_stats = extract_round_stats(run, cumulative_metric, aggs=("mean",))
        if cum_stats.get("mean"):
            has_cumulative = True

    if not has_per_round and not has_cumulative:
        msg = "No privacy metrics (epsilon or rdp_cost) found in any of the specified runs."
        raise ValueError(msg)

    n_subplots = int(has_per_round) + int(has_cumulative)
    fig, axes = _setup_figure(n_subplots, len(run_ids))
    ax_idx = 0

    if has_per_round:
        ax = axes[ax_idx]
        for i, (run_id, label) in enumerate(
            zip(run_ids, resolved_labels, strict=True),
        ):
            run = get_run_by_id(run_id)
            color = palette[i]
            ls = LINE_STYLES[i % len(LINE_STYLES)]

            rdp = run_rdp_flags[run_id]
            per_round_metric = "rdp_cost" if rdp else "epsilon"
            aggs = ("mean", "std") if show_std else ("mean",)
            rounds_per, eps_stats = extract_round_stats(run, per_round_metric, aggs=aggs)
            epsilons = eps_stats.get("mean", [])
            if epsilons:
                ax.plot(
                    rounds_per, epsilons, color=color, linestyle=ls,
                    linewidth=2, label=label,
                )
                if show_std:
                    stds = eps_stats.get("std", [])
                    if stds and len(stds) == len(epsilons):
                        upper = [m + s for m, s in zip(epsilons, stds, strict=True)]
                        lower = [m - s for m, s in zip(epsilons, stds, strict=True)]
                        ax.fill_between(rounds_per, lower, upper, alpha=0.1, color=color)

        any_rdp = any(run_rdp_flags.get(rid, False) for rid in run_ids)
        x_label = "Per-Round RDP(α)" if any_rdp else "Per-Round Epsilon (ε)"
        ax.set_xlabel("Round")
        ax.set_ylabel(x_label)
        ax.set_title("Per-Round Privacy Cost")
        ax.legend(frameon=True, framealpha=0.9, edgecolor="gray")
        ax_idx += 1

    if has_cumulative:
        ax = axes[ax_idx]
        for i, (run_id, label) in enumerate(
            zip(run_ids, resolved_labels, strict=True),
        ):
            run = get_run_by_id(run_id)
            color = palette[i]
            ls = LINE_STYLES[i % len(LINE_STYLES)]

            rdp = run_rdp_flags[run_id]
            cumulative_metric = "cumulative_rdp" if rdp else "cumulative_epsilon"
            rounds_cum, cum_stats = extract_round_stats(
                run, cumulative_metric, aggs=("mean",),
            )
            cum_epsilons = cum_stats.get("mean", [])
            if cum_epsilons:
                ax.plot(
                    rounds_cum, cum_epsilons, color=color, linestyle=ls,
                    linewidth=2, label=label,
                )

        any_rdp = any(run_rdp_flags.get(rid, False) for rid in run_ids)
        y_label = "Cumulative RDP(α)" if any_rdp else "Cumulative Epsilon (ε)"
        ax.set_xlabel("Round")
        ax.set_ylabel(y_label)
        ax.set_title("Cumulative Privacy Cost")
        ax.legend(frameon=True, framealpha=0.9, edgecolor="gray")

    fig.suptitle("Privacy Budget Comparison")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
