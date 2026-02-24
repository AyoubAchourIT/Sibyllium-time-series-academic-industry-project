from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot per-horizon metrics from a run directory.")
    p.add_argument("--run-dir", required=True, help="Path to runs/<RUN_ID>")
    return p


def _load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"metrics.json not found in {run_dir}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _plot_metric(run_dir: Path, metrics: dict, key: str, ylabel: str, output_name: str) -> bool:
    plt.figure(figsize=(10, 5))
    any_series = False
    for model_name, vals in metrics.items():
        if not isinstance(vals, dict):
            continue
        series = vals.get(key)
        if not isinstance(series, list) or len(series) == 0:
            continue
        y = np.asarray(series, dtype=float)
        if y.ndim != 1 or y.size == 0:
            continue
        x = np.arange(1, y.size + 1)
        plt.plot(x, y, marker="o", linewidth=1.5, markersize=3, label=model_name)
        any_series = True

    if not any_series:
        plt.close()
        print(f"No '{key}' arrays found in metrics.json (older run format).")
        return False

    plt.title(f"{ylabel} by Horizon")
    plt.xlabel("Horizon k")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    if key.startswith("trend"):
        plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    out_path = run_dir / output_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return True


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    metrics = _load_metrics(run_dir)
    _plot_metric(run_dir, metrics, "trend_by_horizon", "Trend Accuracy", "horizon_trend.png")
    _plot_metric(run_dir, metrics, "corr_by_horizon", "Correlation", "horizon_corr.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
