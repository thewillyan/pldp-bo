import warnings
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import seaborn as sns

from src.plotting._helpers import (
    extract_metrics_by_round,
    get_run_by_id,
    get_run_name,
)

LINE_STYLES = ["-", "--", "-.", ":"]
PALETTE = "husl"


def _resolve_labels(
    run_ids: Sequence[str],
    labels: Optional[Sequence[str]],
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
    labels: Optional[Sequence[str]] = None,
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> matplotlib.figure.Figure:
    resolved_labels = _resolve_labels(run_ids, labels)
    palette = sns.color_palette(PALETTE, n_colors=len(run_ids))

    fig, (ax_loss, ax_acc) = _setup_figure(2, len(run_ids))

    for i, (run_id, label) in enumerate(
        zip(run_ids, resolved_labels, strict=True)
    ):
        run = get_run_by_id(run_id)
        color = palette[i]
        ls = LINE_STYLES[i % len(LINE_STYLES)]

        rounds, losses = extract_metrics_by_round(run, "server_loss")
        if rounds:
            ax_loss.plot(
                rounds, losses, color=color, linestyle=ls, linewidth=2, label=label
            )

        rounds, accuracies = extract_metrics_by_round(run, "accuracy")
        if rounds:
            ax_acc.plot(
                rounds,
                accuracies,
                color=color,
                linestyle=ls,
                linewidth=2,
                label=label,
            )

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


def plot_comparison_privacy(
    run_ids: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> matplotlib.figure.Figure:
    resolved_labels = _resolve_labels(run_ids, labels)
    palette = sns.color_palette(PALETTE, n_colors=len(run_ids))

    fig, axes = _setup_figure(1, len(run_ids))
    ax = axes[0]

    has_data = False
    for i, (run_id, label) in enumerate(
        zip(run_ids, resolved_labels, strict=True)
    ):
        run = get_run_by_id(run_id)
        color = palette[i]
        ls = LINE_STYLES[i % len(LINE_STYLES)]

        rounds, epsilons = extract_metrics_by_round(run, "epsilon")
        if rounds:
            ax.plot(
                rounds, epsilons, color=color, linestyle=ls, linewidth=2, label=label
            )
            has_data = True

    if not has_data:
        msg = "No epsilon metrics found in any of the specified runs."
        warnings.warn(msg, stacklevel=2)

    ax.set_xlabel("Round")
    ax.set_ylabel("Epsilon (ε)")
    ax.set_title("Privacy Budget (ε) vs Round")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="gray")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
