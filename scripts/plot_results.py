from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import seaborn as sns

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from src.plotting import get_run_name, get_run_params, list_runs
from src.plotting.comparison import (
    plot_comparison_convergence,
    plot_comparison_privacy,
)
from src.plotting.convergence import plot_convergence
from src.plotting.privacy import plot_privacy_budget


def _format_time(start_ms: int, end_ms: int | None) -> str:
    dt_start = datetime.fromtimestamp(start_ms / 1000)
    start_str = dt_start.strftime("%d/%m/%Y %H:%M:%S")

    if end_ms is None:
        return f"{start_str} \u2192 \u2014"

    dt_end = datetime.fromtimestamp(end_ms / 1000)
    end_str = dt_end.strftime("%d/%m/%Y %H:%M:%S")
    elapsed_s = int((end_ms - start_ms) / 1000)

    days, elapsed_s = divmod(elapsed_s, 86400)
    hours, elapsed_s = divmod(elapsed_s, 3600)
    minutes, seconds = divmod(elapsed_s, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")

    return f"{start_str} \u2192 {end_str} ({' '.join(parts)})"


def _set_tracking_uri(args: argparse.Namespace) -> None:
    if args.tracking_uri is not None:
        mlflow.set_tracking_uri(args.tracking_uri)


def cmd_list_runs(args: argparse.Namespace) -> None:
    _set_tracking_uri(args)
    runs = list_runs(experiment_name=args.experiment)
    if not runs:
        print("No runs found.")
        return

    header = (
        f"{'Run ID':<10} {'Name':<25} {'Strategy':<12}"
        f" {'Rounds':<8} {'DP':<5} {'Time':<50}"
    )
    print(header)
    print("-" * len(header))

    for run in runs:
        short_id = run.info.run_id[:8]
        name = get_run_name(run)
        params = get_run_params(run)
        strategy = params.get("federated.strategy", "?")
        rounds = params.get("federated.num_rounds", "?")
        dp = "yes" if params.get("privacy.enabled") == "True" else "no"
        time_col = _format_time(run.info.start_time, run.info.end_time)
        print(
            f"{short_id:<10} {name:<25} {strategy:<12}"
            f" {rounds:<8} {dp:<5} {time_col:<50}"
        )


def cmd_single(args: argparse.Namespace) -> None:
    _set_tracking_uri(args)
    if plt is None:
        print("Error: matplotlib is required for plotting", file=sys.stderr)
        sys.exit(1)
    _set_plot_theme()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.type in ("convergence", "all"):
        path = save_dir / "convergence.png"
        fig = plot_convergence(args.run_id, save_path=path)
        print(f"Saved convergence plot to {path}")
        plt.close(fig)

    if args.type in ("privacy", "all"):
        path = save_dir / "privacy_budget.png"
        try:
            fig = plot_privacy_budget(args.run_id, save_path=path)
            print(f"Saved privacy plot to {path}")
            plt.close(fig)
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)


def cmd_compare(args: argparse.Namespace) -> None:
    _set_tracking_uri(args)
    if plt is None:
        print("Error: matplotlib is required for plotting", file=sys.stderr)
        sys.exit(1)
    _set_plot_theme()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_ids = args.runs
    labels = args.names

    if labels and len(labels) != len(run_ids):
        msg = (
            f"Error: --names requires exactly {len(run_ids)} labels,"
            f" got {len(labels)}"
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

    if args.type in ("convergence", "all"):
        path = save_dir / "convergence.png"
        fig = plot_comparison_convergence(
            run_ids, labels=labels, save_path=path
        )
        print(f"Saved comparison convergence plot to {path}")
        plt.close(fig)

    if args.type in ("privacy", "all"):
        path = save_dir / "privacy_budget.png"
        try:
            fig = plot_comparison_privacy(
                run_ids, labels=labels, save_path=path
            )
            print(f"Saved comparison privacy plot to {path}")
            plt.close(fig)
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)


def cmd_get_run_id(args: argparse.Namespace) -> None:
    _set_tracking_uri(args)
    client = mlflow.tracking.MlflowClient()
    experiment_ids = [exp.experiment_id for exp in client.search_experiments()]
    safe_name = args.run_name.replace("'", "\\'")
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=f"attributes.run_name = '{safe_name}'",
        order_by=["start_time DESC"],
    )
    if not runs:
        print(f"Run '{args.run_name}' not found", file=sys.stderr)
        sys.exit(1)
    print(runs[0].info.run_id)


def _set_plot_theme() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)


def main() -> None:
    parser = argparse.ArgumentParser(description="PLDP-BO plot tools")
    parser.add_argument(
        "--tracking-uri", type=str, default=None,
        help="MLflow tracking URI (overrides default)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-runs", help="List available runs")
    list_parser.add_argument(
        "--experiment", type=str, default=None, help="Filter by experiment name"
    )

    get_run_id_parser = subparsers.add_parser(
        "get-run-id", help="Print run ID for a given run name"
    )
    get_run_id_parser.add_argument(
        "--run-name", type=str, required=True, help="MLflow run name"
    )

    single_parser = subparsers.add_parser("plot", help="Plot a single run")
    single_parser.add_argument("run_id", type=str, help="Run UUID")
    single_parser.add_argument(
        "--type",
        choices=["convergence", "privacy", "all"],
        default="all",
    )
    single_parser.add_argument("--save-dir", type=str, default="./plots")

    compare_parser = subparsers.add_parser("compare", help="Compare multiple runs")
    compare_parser.add_argument(
        "--runs", nargs="+", required=True, help="Run UUIDs to compare"
    )
    compare_parser.add_argument(
        "--names", nargs="+", default=None, help="Labels for each run"
    )
    compare_parser.add_argument(
        "--type",
        choices=["convergence", "privacy", "all"],
        default="all",
    )
    compare_parser.add_argument("--save-dir", type=str, default="./plots")

    args = parser.parse_args()

    if args.command == "list-runs":
        cmd_list_runs(args)
    elif args.command == "get-run-id":
        cmd_get_run_id(args)
    elif args.command == "plot":
        cmd_single(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
