from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASELINE_KEYS = {"persistence", "drift"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a benchmark registry table from run artifacts.")
    p.add_argument("--runs-dir", default="runs", help="Directory containing runs/*")
    p.add_argument("--out", default=None, help="Output CSV path (default: runs/benchmark_table.csv)")
    return p


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_n_val(run_dir: Path, config: dict[str, Any]) -> int | None:
    p = run_dir / "y_val.npy"
    if p.exists():
        try:
            arr = np.load(p)
            return int(arr.shape[0]) if arr.ndim >= 1 else 0
        except Exception:
            return None
    if isinstance(config.get("rows_val"), int):
        return int(config["rows_val"])
    return None


def _main_metric_entry(config: dict[str, Any], metrics: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    model = str(config.get("model", "")).lower() if config else ""
    if model in metrics and isinstance(metrics.get(model), dict):
        return model, metrics[model]
    for k, v in metrics.items():
        if k in BASELINE_KEYS:
            continue
        if isinstance(v, dict) and "composite" in v:
            return k, v
    return None, None


def _band_mean(vals: Any, start: int, end: int) -> float | None:
    if not isinstance(vals, list) or not vals:
        return None
    a = np.asarray(vals, dtype=float)
    if a.ndim != 1 or a.size == 0:
        return None
    s = max(1, start)
    e = min(int(a.size), end)
    if s > e:
        return None
    return float(a[s - 1 : e].mean())


def build_benchmark_table(runs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return pd.DataFrame()
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        config = _load_json(run_dir / "config.json") or {}
        metrics = _load_json(run_dir / "metrics.json") or {}
        if not metrics:
            continue
        model_key, m = _main_metric_entry(config, metrics)
        if not model_key or not isinstance(m, dict):
            continue
        row = {
            "run_id": run_dir.name,
            "target": config.get("target"),
            "model": model_key,
            "delta_target": config.get("delta_target"),
            "lookback": config.get("lookback"),
            "lookback_lags": config.get("lookback_lags"),
            "limit_files": config.get("limit_files"),
            "n_val": _load_n_val(run_dir, config),
            "trend": m.get("trend_accuracy"),
            "corr": m.get("mean_correlation"),
            "composite": m.get("composite"),
            "trend_k1_5": _band_mean(m.get("trend_by_horizon"), 1, 5),
            "trend_k11_15": _band_mean(m.get("trend_by_horizon"), 11, 15),
            "corr_k1_5": _band_mean(m.get("corr_by_horizon"), 1, 5),
            "corr_k11_15": _band_mean(m.get("corr_by_horizon"), 11, 15),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "composite" in df.columns:
        df["composite"] = pd.to_numeric(df["composite"], errors="coerce")
        df = df.sort_values(["composite", "run_id"], ascending=[False, False], na_position="last").reset_index(drop=True)
    return df


def main() -> int:
    args = build_parser().parse_args()
    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out) if args.out else (runs_dir / "benchmark_table.csv")
    df = build_benchmark_table(runs_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if df.empty:
        print("No valid run entries found.")
    else:
        cols = [
            "run_id",
            "target",
            "model",
            "delta_target",
            "lookback",
            "lookback_lags",
            "limit_files",
            "n_val",
            "trend",
            "corr",
            "composite",
        ]
        print(df[cols].to_string(index=False))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
