from __future__ import annotations

import argparse

import numpy as np

from src.metrics.forecast import composite_score, mean_horizon_correlation, trend_accuracy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Sibyllium forecasts (scaffold demo).")
    parser.add_argument("--demo", action="store_true", help="Run a small built-in metric demo")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # Default behavior is demo mode for a minimal runnable entrypoint.
    if args.demo or True:
        y_current = np.array([10.0, 5.0])
        y_true = np.array([[10.2, 10.1, 10.5], [4.9, 4.8, 5.1]])
        y_pred = np.array([[10.1, 10.0, 10.4], [4.8, 4.9, 5.0]])
        trend = trend_accuracy(y_true, y_pred, y_current)
        corr = mean_horizon_correlation(y_true, y_pred)
        score = composite_score(y_true, y_pred, y_current)
        print(f"trend_accuracy={trend:.4f}")
        print(f"mean_horizon_correlation={corr:.4f}")
        print(f"composite_score={score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
