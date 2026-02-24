from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASELINE_KEYS = {"persistence", "drift"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize recent run artifacts.")
    p.add_argument("--runs-dir", default="runs", help="Directory containing run folders")
    p.add_argument("--last", type=int, default=20, help="Number of most recent run folders to summarize")
    return p


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _infer_metric_key(config: dict[str, Any], metrics: dict[str, Any]) -> str | None:
    model = str(config.get("model", "")).lower() if config else ""
    if model in metrics and isinstance(metrics.get(model), dict):
        return model
    # Fall back to best non-baseline metric entry.
    candidates = []
    for k, v in metrics.items():
        if k in BASELINE_KEYS or not isinstance(v, dict):
            continue
        comp = v.get("composite")
        if isinstance(comp, (int, float)):
            candidates.append((float(comp), k))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def _load_n_val(run_dir: Path, config: dict[str, Any]) -> int | None:
    y_val_path = run_dir / "y_val.npy"
    if y_val_path.exists():
        try:
            arr = np.load(y_val_path)
            return int(arr.shape[0]) if arr.ndim >= 1 else 0
        except Exception:
            pass
    rows_val = config.get("rows_val") if config else None
    if isinstance(rows_val, int):
        return rows_val
    return None


def _val_fraction(config: dict[str, Any]) -> float | None:
    if not config:
        return None
    rows_total = config.get("rows_total")
    rows_val = config.get("rows_val")
    if isinstance(rows_total, int) and rows_total > 0 and isinstance(rows_val, int):
        return float(rows_val) / float(rows_total)
    return None


def _run_dirs(runs_dir: Path, last: int) -> list[Path]:
    dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    dirs = sorted(dirs, key=lambda p: p.name, reverse=True)
    if last < 1:
        return []
    return dirs[:last]


def summarize_runs(runs_dir: Path, last: int = 20) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in _run_dirs(runs_dir, last):
        config = _load_json(run_dir / "config.json") or {}
        metrics = _load_json(run_dir / "metrics.json") or {}
        if not metrics:
            continue
        metric_key = _infer_metric_key(config, metrics)
        if metric_key is None:
            continue
        m = metrics.get(metric_key, {})
        if not isinstance(m, dict):
            continue
        row = {
            "run_id": run_dir.name,
            "target": config.get("target"),
            "model": metric_key,
            "trend": m.get("trend_accuracy"),
            "corr": m.get("mean_correlation"),
            "composite": m.get("composite"),
            "n_val": _load_n_val(run_dir, config),
            "limit_files": config.get("limit_files"),
            "delta_target": config.get("delta_target"),
            "lookback_lags": config.get("lookback_lags"),
            "val_fraction": _val_fraction(config),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "target",
                "model",
                "trend",
                "corr",
                "composite",
                "n_val",
                "limit_files",
                "delta_target",
                "lookback_lags",
                "val_fraction",
            ]
        )
    for col in ["trend", "corr", "composite", "val_fraction"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("composite", ascending=False, na_position="last").reset_index(drop=True)
    return df


def main() -> int:
    args = build_parser().parse_args()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {runs_dir}")
    df = summarize_runs(runs_dir, last=args.last)
    out_path = runs_dir / f"summary_last{args.last}.csv"
    df.to_csv(out_path, index=False)
    if df.empty:
        print("No valid runs found.")
        print(f"Saved {out_path}")
        return 0

    display_cols = [
        "run_id",
        "target",
        "model",
        "trend",
        "corr",
        "composite",
        "n_val",
        "limit_files",
        "delta_target",
        "lookback_lags",
        "val_fraction",
    ]
    print(df[display_cols].to_string(index=False))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
