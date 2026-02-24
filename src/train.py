from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io import find_excel_files, infer_symbol_from_path, load_excel_ohlcv
from src.features.tabular import make_supervised_multi_horizon
from src.features.technical import calculate_macd, calculate_stochastic
from src.features.windowing import make_windows_univariate
from src.metrics.forecast import composite_score, mean_horizon_correlation, trend_accuracy
from src.models.baselines import drift_forecast, persistence_forecast
from src.models.sklearn_models import predict_multioutput, train_multioutput_gbdt

TARGET_CHOICES = (
    "macd",
    "macd_signal",
    "macd_histogram",
    "stoch_k",
    "stoch_d",
    "stoch_d_smooth",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Sibyllium forecasting models.")
    parser.add_argument("--data-dir", default="datas", help="Directory containing XLSX files")
    parser.add_argument("--runs-dir", default="runs", help="Directory to store run artifacts")
    parser.add_argument("--horizon", type=int, default=15, help="Forecast horizon in bars")
    parser.add_argument("--limit-files", type=int, default=3, help="Max files to process (default: 3)")
    parser.add_argument("--symbol-pattern", default=None, help="Optional substring filter on symbol/file name")
    parser.add_argument("--target", choices=TARGET_CHOICES, default="macd", help="Indicator target column")
    parser.add_argument("--lookback-lags", type=int, default=64, help="Number of lag features (1..N) for GBDT")
    parser.add_argument("--delta-target", action="store_true", help="Train GBDT on target deltas (Y - y0)")
    parser.add_argument("--sweep", action="store_true", help="Run a small GBDT hyperparameter sweep")
    parser.add_argument("--model", choices=("gbdt", "dlinear"), default="gbdt", help="Main model to train")
    parser.add_argument("--lookback", type=int, default=256, help="Sequence lookback for DLinear")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for DLinear")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for DLinear")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for DLinear")
    return parser


def _select_files(data_dir: str | Path, limit_files: int, symbol_pattern: str | None) -> list[Path]:
    files = find_excel_files(data_dir)
    if symbol_pattern:
        p = symbol_pattern.lower()
        files = [f for f in files if p in f.name.lower() or p in infer_symbol_from_path(f).lower()]
    if limit_files < 1:
        return []
    return files[:limit_files]


def _roll_windows_from_lookback(lookback_lags: int) -> list[int]:
    candidates = [3, 5, 10, 20, 50]
    windows = sorted({w for w in candidates if 1 <= w <= lookback_lags})
    return windows or [max(1, lookback_lags)]


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, y0: np.ndarray) -> dict[str, float]:
    corr = mean_horizon_correlation(y_true, y_pred)
    return {
        "trend_accuracy": float(trend_accuracy(y_true, y_pred, y0)),
        "mean_correlation": float(corr),
        "composite": float(composite_score(y_true, y_pred, y0)),
    }


def _load_with_indicators(path: Path) -> pd.DataFrame:
    ohlcv = load_excel_ohlcv(path)
    indicators = pd.concat([calculate_macd(ohlcv), calculate_stochastic(ohlcv)], axis=1)
    return pd.concat([ohlcv, indicators], axis=1)


def _prepare_indicator_cache(selected_files: list[Path]) -> list[dict[str, object]]:
    """Load OHLCV + indicators once per file and cache for reuse (e.g., sweep mode)."""
    cache: list[dict[str, object]] = []
    for path in selected_files:
        symbol = infer_symbol_from_path(path)
        try:
            df = _load_with_indicators(path)
            cache.append({"path": path, "symbol": symbol, "df": df, "error": None})
        except Exception as exc:
            cache.append({"path": path, "symbol": symbol, "df": None, "error": str(exc)})
    return cache


def _assemble_tabular_dataset_from_cache(
    cache: list[dict[str, object]],
    *,
    target: str,
    horizon: int,
    lookback_lags: int,
    roll_windows: list[int] | None = None,
) -> dict[str, object]:
    lags = list(range(1, lookback_lags + 1))
    rw = list(roll_windows) if roll_windows is not None else _roll_windows_from_lookback(lookback_lags)

    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.DataFrame] = []
    y0_parts: list[pd.Series] = []
    drift_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []
    file_summaries: list[dict[str, object]] = []
    skipped_files = 0

    for rec in cache:
        path = rec["path"]
        symbol = rec["symbol"]
        df = rec["df"]
        err = rec["error"]
        assert isinstance(path, Path)
        assert isinstance(symbol, str)
        if err is not None or df is None:
            skipped_files += 1
            file_summaries.append({"symbol": symbol, "file": path.name, "error": str(err)})
            continue
        assert isinstance(df, pd.DataFrame)
        try:
            X_i, Y_i, y0_i = make_supervised_multi_horizon(
                df=df,
                target_col=target,
                horizon=horizon,
                lags=lags,
                roll_windows=rw,
            )
            if X_i.empty:
                skipped_files += 1
                file_summaries.append({"symbol": symbol, "file": path.name, "rows_supervised": 0})
                continue
            x_parts.append(X_i)
            y_parts.append(Y_i)
            y0_parts.append(y0_i)
            drift_parts.append(drift_forecast(y0_i.to_numpy(dtype=float), horizon=horizon))
            meta_parts.append(pd.DataFrame({"symbol": [symbol] * len(X_i), "source_file": [path.name] * len(X_i)}))
            file_summaries.append(
                {"symbol": symbol, "file": path.name, "rows_ohlcv": int(len(df)), "rows_supervised": int(len(X_i))}
            )
        except Exception as exc:
            skipped_files += 1
            file_summaries.append({"symbol": symbol, "file": path.name, "error": str(exc)})

    if not x_parts:
        raise SystemExit("No usable supervised rows were produced from selected files.")

    return {
        "X": pd.concat(x_parts, ignore_index=True),
        "Y": pd.concat(y_parts, ignore_index=True),
        "y0": pd.concat(y0_parts, ignore_index=True),
        "drift_pred": np.vstack(drift_parts).astype(float, copy=False),
        "meta": pd.concat(meta_parts, ignore_index=True),
        "lags": lags,
        "roll_windows": rw,
        "file_summaries": file_summaries,
        "processed_files": len(x_parts),
        "skipped_files": skipped_files,
    }


def _assemble_tabular_dataset(args: argparse.Namespace, selected_files: list[Path]) -> dict[str, object]:
    cache = _prepare_indicator_cache(selected_files)
    return _assemble_tabular_dataset_from_cache(
        cache,
        target=args.target,
        horizon=args.horizon,
        lookback_lags=args.lookback_lags,
    )


def _assemble_dlinear_dataset(args: argparse.Namespace, selected_files: list[Path]) -> dict[str, object]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    y0_parts: list[np.ndarray] = []
    drift_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []
    file_summaries: list[dict[str, object]] = []
    skipped_files = 0

    for path in selected_files:
        symbol = infer_symbol_from_path(path)
        try:
            df = _load_with_indicators(path)
            X_i, Y_i, y0_i = make_windows_univariate(df[args.target], lookback=args.lookback, horizon=args.horizon)
            if X_i.shape[0] == 0:
                skipped_files += 1
                file_summaries.append({"symbol": symbol, "file": path.name, "rows_windows": 0})
                continue
            x_parts.append(X_i)
            y_parts.append(Y_i)
            y0_parts.append(y0_i)
            drift_parts.append(drift_forecast(y0_i, horizon=args.horizon))
            meta_parts.append(pd.DataFrame({"symbol": [symbol] * len(X_i), "source_file": [path.name] * len(X_i)}))
            file_summaries.append(
                {
                    "symbol": symbol,
                    "file": path.name,
                    "rows_ohlcv": int(len(df)),
                    "rows_windows": int(X_i.shape[0]),
                }
            )
        except Exception as exc:
            skipped_files += 1
            file_summaries.append({"symbol": symbol, "file": path.name, "error": str(exc)})

    if not x_parts:
        raise SystemExit("No usable DLinear windows were produced from selected files.")

    X_all = np.vstack(x_parts).astype(np.float32, copy=False)
    Y_all = np.vstack(y_parts).astype(np.float32, copy=False)
    y0_all = np.concatenate(y0_parts).astype(float, copy=False)
    drift_all = np.vstack(drift_parts).astype(float, copy=False)
    meta_all = pd.concat(meta_parts, ignore_index=True)

    return {
        "X": X_all,
        "Y": Y_all,
        "y0": y0_all,
        "drift_pred": drift_all,
        "meta": meta_all,
        "file_summaries": file_summaries,
        "processed_files": len(x_parts),
        "skipped_files": skipped_files,
    }


def _run_gbdt_experiment(
    ds: dict[str, object],
    *,
    horizon: int,
    delta_target: bool,
) -> dict[str, object]:
    """Train/evaluate baselines + GBDT on a prebuilt tabular dataset."""
    X_all = ds["X"]
    Y_all = ds["Y"]
    y0_all = ds["y0"]
    drift_all = ds["drift_pred"]
    assert isinstance(X_all, pd.DataFrame)
    assert isinstance(Y_all, pd.DataFrame)
    assert isinstance(y0_all, pd.Series)
    n_rows = len(X_all)
    if n_rows < 2:
        raise SystemExit("Need at least 2 rows/windows to create train/validation split.")
    split_idx = max(1, min(n_rows - 1, int(n_rows * 0.8)))

    X_train, X_val = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    Y_train, Y_val = Y_all.iloc[:split_idx], Y_all.iloc[split_idx:]
    y0_train_np = y0_all.iloc[:split_idx].to_numpy(dtype=float)
    y0_val_np = y0_all.iloc[split_idx:].to_numpy(dtype=float)
    y_val_np = Y_val.to_numpy(dtype=float)

    pred_persistence = persistence_forecast(y0_val_np, horizon=horizon)
    pred_drift = np.asarray(drift_all[split_idx:], dtype=float)

    metrics: dict[str, dict[str, float]] = {
        "persistence": _evaluate(y_val_np, pred_persistence, y0_val_np),
        "drift": _evaluate(y_val_np, pred_drift, y0_val_np),
    }

    pred_model_delta = None
    if delta_target:
        train_target = Y_train.to_numpy(dtype=float) - y0_train_np.reshape(-1, 1)
        models = train_multioutput_gbdt(X_train, train_target)
        pred_model_delta = predict_multioutput(models, X_val)
        pred_model = pred_model_delta + y0_val_np.reshape(-1, 1)
    else:
        models = train_multioutput_gbdt(X_train, Y_train)
        pred_model = predict_multioutput(models, X_val)
    metrics["gbdt"] = _evaluate(y_val_np, pred_model, y0_val_np)

    return {
        "metrics": metrics,
        "pred_model": np.asarray(pred_model, dtype=float),
        "pred_model_delta": None if pred_model_delta is None else np.asarray(pred_model_delta, dtype=float),
        "pred_persistence": pred_persistence,
        "pred_drift": pred_drift,
        "y_val": y_val_np,
        "y0_val": y0_val_np,
        "split_idx": split_idx,
        "n_rows": n_rows,
        "n_features": int(X_train.shape[1]),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.horizon < 1:
        raise SystemExit("--horizon must be >= 1")
    if args.lookback_lags < 1:
        raise SystemExit("--lookback-lags must be >= 1")
    if args.lookback < 1:
        raise SystemExit("--lookback must be >= 1")

    selected_files = _select_files(args.data_dir, args.limit_files, args.symbol_pattern)
    if not selected_files:
        raise SystemExit("No Excel files found after applying filters/limit.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        if args.model != "gbdt":
            raise SystemExit("--sweep currently supports only --model gbdt")
        cache = _prepare_indicator_cache(selected_files)
        grid_delta = [False, True]
        grid_lags = [32, 64, 128]
        grid_roll_windows = [[5, 10, 20], [5, 10, 20, 50]]
        rows: list[dict[str, object]] = []
        print("Running GBDT sweep...")
        for delta_target in grid_delta:
            for lookback_lags in grid_lags:
                for roll_windows in grid_roll_windows:
                    try:
                        ds_sweep = _assemble_tabular_dataset_from_cache(
                            cache,
                            target=args.target,
                            horizon=args.horizon,
                            lookback_lags=lookback_lags,
                            roll_windows=roll_windows,
                        )
                        result = _run_gbdt_experiment(ds_sweep, horizon=args.horizon, delta_target=delta_target)
                        m = result["metrics"]["gbdt"]
                        row = {
                            "delta_target": delta_target,
                            "lookback_lags": lookback_lags,
                            "roll_windows": "|".join(map(str, roll_windows)),
                            "rows_total": int(result["n_rows"]),
                            "n_features": int(result["n_features"]),
                            "trend_accuracy": float(m["trend_accuracy"]),
                            "mean_correlation": float(m["mean_correlation"]),
                            "composite": float(m["composite"]),
                            "processed_files": int(ds_sweep["processed_files"]),
                            "skipped_files": int(ds_sweep["skipped_files"]),
                        }
                        print(
                            "sweep "
                            f"delta={int(delta_target)} lags={lookback_lags:3d} "
                            f"rolls={row['roll_windows']:>10s} "
                            f"comp={row['composite']:.4f} trend={row['trend_accuracy']:.4f} "
                            f"corr={row['mean_correlation']:.4f}"
                        )
                    except (Exception, SystemExit) as exc:  # keep sweep moving on bad configs/files
                        row = {
                            "delta_target": delta_target,
                            "lookback_lags": lookback_lags,
                            "roll_windows": "|".join(map(str, roll_windows)),
                            "error": str(exc),
                        }
                        print(
                            "sweep "
                            f"delta={int(delta_target)} lags={lookback_lags:3d} "
                            f"rolls={'|'.join(map(str, roll_windows)):>10s} error={exc}"
                        )
                    rows.append(row)

        sweep_df = pd.DataFrame(rows)
        sweep_df.to_csv(run_dir / "sweep_results.csv", index=False)
        config = {
            **vars(args),
            "run_id": run_id,
            "selected_files": [str(p) for p in selected_files],
            "sweep_grid": {
                "delta_target": grid_delta,
                "lookback_lags": grid_lags,
                "roll_windows": grid_roll_windows,
            },
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Run artifacts: {run_dir}")
        print(f"Sweep results saved: {run_dir / 'sweep_results.csv'}")
        return 0

    if args.model == "gbdt":
        ds = _assemble_tabular_dataset(args, selected_files)
        X_all = ds["X"]
        Y_all = ds["Y"]
        y0_all = ds["y0"]
        drift_all = ds["drift_pred"]
        meta_all = ds["meta"]
        n_rows = len(X_all)
    else:
        ds = _assemble_dlinear_dataset(args, selected_files)
        X_all = ds["X"]
        Y_all = ds["Y"]
        y0_all = ds["y0"]
        drift_all = ds["drift_pred"]
        meta_all = ds["meta"]
        n_rows = int(X_all.shape[0])

    if args.model == "gbdt":
        result = _run_gbdt_experiment(ds, horizon=args.horizon, delta_target=bool(args.delta_target))
        pred_model = result["pred_model"]
        pred_model_delta = result["pred_model_delta"]
        pred_persistence = result["pred_persistence"]
        pred_drift = result["pred_drift"]
        y_val_np = result["y_val"]
        y0_val_np = result["y0_val"]
        split_idx = int(result["split_idx"])
        n_rows = int(result["n_rows"])
        metrics: dict[str, dict[str, float]] = result["metrics"]
        np.save(run_dir / "predictions_val.npy", pred_model)
        np.save(run_dir / "predictions_val_gbdt.npy", pred_model)
        if pred_model_delta is not None:
            np.save(run_dir / "predictions_val_gbdt_delta.npy", pred_model_delta)
        n_features = int(result["n_features"])
    else:
        if n_rows < 2:
            raise SystemExit("Need at least 2 rows/windows to create train/validation split.")
        split_idx = max(1, min(n_rows - 1, int(n_rows * 0.8)))
        X_train, X_val = X_all[:split_idx], X_all[split_idx:]
        Y_train, Y_val = Y_all[:split_idx], Y_all[split_idx:]
        y0_val_np = np.asarray(y0_all[split_idx:], dtype=float)
        y_val_np = np.asarray(Y_val, dtype=float)
        pred_persistence = persistence_forecast(y0_val_np, horizon=args.horizon)
        pred_drift = np.asarray(drift_all[split_idx:], dtype=float)
        metrics = {
            "persistence": _evaluate(y_val_np, pred_persistence, y0_val_np),
            "drift": _evaluate(y_val_np, pred_drift, y0_val_np),
        }
        try:
            from src.models.dlinear import predict_dlinear, train_dlinear
        except ImportError as exc:
            raise SystemExit("DLinear requested but torch is not installed. Install requirements.txt first.") from exc

        model = train_dlinear(
            X_train,
            Y_train,
            X_val,
            Y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        pred_model = predict_dlinear(model, X_val)
        metrics["dlinear"] = _evaluate(y_val_np, pred_model, y0_val_np)
        np.save(run_dir / "predictions_val.npy", pred_model)
        np.save(run_dir / "predictions_val_dlinear.npy", pred_model)
        n_features = int(X_train.shape[1])

    np.save(run_dir / "predictions_val_persistence.npy", pred_persistence)
    np.save(run_dir / "predictions_val_drift.npy", pred_drift)
    np.save(run_dir / "y_val.npy", y_val_np)
    np.save(run_dir / "y0_val.npy", y0_val_np)

    meta_val = meta_all.iloc[split_idx:].reset_index(drop=True)
    model_key = args.model
    preview = pd.DataFrame(
        {
            "symbol": meta_val["symbol"].to_numpy(),
            "y0": y0_val_np,
            "y_true_t+1": y_val_np[:, 0],
            "y_pred_persistence_t+1": pred_persistence[:, 0],
            "y_pred_drift_t+1": pred_drift[:, 0],
            f"y_pred_{model_key}_t+1": pred_model[:, 0],
        }
    )
    for k in range(2, min(args.horizon, 3) + 1):
        preview[f"y_true_t+{k}"] = y_val_np[:, k - 1]
        preview[f"y_pred_{model_key}_t+{k}"] = pred_model[:, k - 1]
    preview.head(200).to_csv(run_dir / "preview_val.csv", index=False)

    config = {
        **vars(args),
        "run_id": run_id,
        "selected_files": [str(p) for p in selected_files],
        "processed_files": int(ds["processed_files"]),
        "skipped_files": int(ds["skipped_files"]),
        "rows_total": int(n_rows),
        "rows_train": int(split_idx),
        "rows_val": int(n_rows - split_idx),
        "n_features": n_features,
        "file_summaries": ds["file_summaries"],
    }
    if args.model == "gbdt":
        config["lags_count"] = len(ds["lags"])
        config["roll_windows"] = ds["roll_windows"]

    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Run artifacts: {run_dir}")
    print(
        f"Processed {ds['processed_files']}/{len(selected_files)} files "
        f"(skipped: {ds['skipped_files']}), rows={n_rows}, train={split_idx}, val={n_rows - split_idx}"
    )
    print("Metrics:")
    for name, vals in metrics.items():
        print(
            f"  {name:12s} trend={vals['trend_accuracy']:.4f} "
            f"corr={vals['mean_correlation']:.4f} composite={vals['composite']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
