from __future__ import annotations

import argparse
from pathlib import Path

from src.plotting.convergence import plot_convergence
from src.plotting.privacy import plot_privacy_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--type", type=str, choices=["convergence", "privacy", "all"], default="all")
    parser.add_argument("--save-dir", type=str, default="./plots")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.type in ("convergence", "all"):
        fig = plot_convergence(args.experiment_id, save_path=save_dir / "convergence.png")
        print(f"Saved convergence plot to {save_dir / 'convergence.png'}")

    if args.type in ("privacy", "all"):
        fig = plot_privacy_budget(args.experiment_id, save_path=save_dir / "privacy_budget.png")
        print(f"Saved privacy plot to {save_dir / 'privacy_budget.png'}")


if __name__ == "__main__":
    main()
