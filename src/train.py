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
from src.metrics.forecast import (
    composite_score,
    correlation_by_horizon,
    mean_horizon_correlation,
    trend_accuracy,
    trend_accuracy_by_horizon,
)
from src.models.baselines import drift_forecast, persistence_forecast
from src.models.ensembles import average_ensemble
from src.models.lightgbm_models import predict_multioutput_lightgbm, train_multioutput_lightgbm
from src.models.sklearn_models import predict_multioutput, train_multioutput_gbdt
from src.models.xgboost_models import predict_multioutput_xgboost, train_multioutput_xgboost

TARGET_CHOICES = (
    "macd",
    "macd_signal",
    "macd_histogram",
    "stoch_k",
    "stoch_d",
    "stoch_d_smooth",
)
STABLE_MODELS = {"gbdt", "dlinear", "lightgbm", "xgboost"}
EXPERIMENTAL_MODELS = {"nhits", "nbeats", "tide", "ensemble_gbdt_nhits"}
ALL_MODELS = STABLE_MODELS | EXPERIMENTAL_MODELS


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
    parser.add_argument("--model", default="gbdt", help="Main model to train (stable: gbdt, dlinear, lightgbm, xgboost)")
    parser.add_argument(
        "--experimental-neuralforecast",
        action="store_true",
        help="Enable experimental NeuralForecast/ensemble models (nhits, nbeats, tide, ensemble_gbdt_nhits)",
    )
    parser.add_argument("--lookback", type=int, default=256, help="Sequence lookback for DLinear")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for DLinear")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for DLinear")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for DLinear")
    parser.add_argument("--input-size", type=int, default=256, help="Input size for NeuralForecast models")
    parser.add_argument("--max-steps", type=int, default=500, help="Max training steps for NeuralForecast models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for supported models")
    parser.add_argument("--val-size", type=int, default=120, help="Validation tail size per series for NF backtesting")
    parser.add_argument("--step-size", type=int, default=1, help="Step size for NF rolling backtest origins")
    parser.add_argument("--allow-small-val", action="store_true", help="Allow NF evaluation with fewer than 200 origins")
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
    return _evaluate_with_corr_basis(y_true, y_pred, y0, use_delta_corr=False)


def _safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else 0.0


def _corr_variants_by_horizon(y_true: np.ndarray, y_pred: np.ndarray, y0: np.ndarray) -> tuple[list[float], list[float]]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    yc = np.asarray(y0, dtype=float).reshape(-1)
    corr_level = [_safe_corr_1d(yt[:, k], yp[:, k]) for k in range(yt.shape[1])]
    dtrue = yt - yc.reshape(-1, 1)
    dpred = yp - yc.reshape(-1, 1)
    corr_delta = [_safe_corr_1d(dtrue[:, k], dpred[:, k]) for k in range(yt.shape[1])]
    return corr_level, corr_delta


def _evaluate_with_corr_basis(
    y_true: np.ndarray, y_pred: np.ndarray, y0: np.ndarray, *, use_delta_corr: bool
) -> dict[str, float]:
    trend_h = trend_accuracy_by_horizon(y_true, y_pred, y0)
    corr_level_h, corr_delta_h = _corr_variants_by_horizon(y_true, y_pred, y0)
    corr_basis = "delta" if use_delta_corr else "level"
    corr_h = corr_delta_h if use_delta_corr else corr_level_h
    mean_corr = float(np.mean(corr_h)) if corr_h else 0.0
    trend = float(trend_accuracy(y_true, y_pred, y0))
    composite = float(0.5 * trend + 0.5 * mean_corr)
    return {
        "trend_accuracy": trend,
        "mean_correlation": mean_corr,
        "composite": composite,
        "trend_by_horizon": [float(v) for v in trend_h],
        "corr_by_horizon": [float(v) for v in corr_h],
        "corr_basis": corr_basis,
        "corr_level_by_horizon": [float(v) for v in corr_level_h],
        "corr_delta_by_horizon": [float(v) for v in corr_delta_h],
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
            X_i, Y_i, y0_i, origin_idx_i = make_supervised_multi_horizon(
                df=df,
                target_col=target,
                horizon=horizon,
                lags=lags,
                roll_windows=rw,
                return_origin_index=True,
            )
            if X_i.empty:
                skipped_files += 1
                file_summaries.append({"symbol": symbol, "file": path.name, "rows_supervised": 0})
                continue
            x_parts.append(X_i)
            y_parts.append(Y_i)
            y0_parts.append(y0_i)
            drift_parts.append(drift_forecast(y0_i.to_numpy(dtype=float), horizon=horizon))
            meta_parts.append(
                pd.DataFrame(
                    {
                        "symbol": [symbol] * len(X_i),
                        "source_file": [path.name] * len(X_i),
                        "origin_ds": pd.to_datetime(origin_idx_i),
                    }
                )
            )
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


def _assemble_nhits_dataset(args: argparse.Namespace, selected_files: list[Path]) -> dict[str, object]:
    from src.models.neuralforecast_models import build_long_df, split_long_df

    per_file_series: list[tuple[str, pd.Series]] = []
    file_summaries: list[dict[str, object]] = []
    skipped_files = 0
    processed_files = 0

    for path in selected_files:
        symbol = infer_symbol_from_path(path)
        try:
            df = _load_with_indicators(path)
            series = df[args.target].dropna()
            if series.empty:
                skipped_files += 1
                file_summaries.append({"symbol": symbol, "file": path.name, "rows_series": 0})
                continue
            per_file_series.append((symbol, series))
            processed_files += 1
            file_summaries.append(
                {"symbol": symbol, "file": path.name, "rows_ohlcv": int(len(df)), "rows_series": int(len(series))}
            )
        except Exception as exc:
            skipped_files += 1
            file_summaries.append({"symbol": symbol, "file": path.name, "error": str(exc)})

    if not per_file_series:
        raise SystemExit("No usable target series were produced from selected files.")

    full_df = build_long_df(per_file_series)
    train_df, val_df = split_long_df(full_df, val_fraction=0.2)
    if train_df.empty or val_df.empty:
        raise SystemExit("NeuralForecast split produced empty train or validation data.")

    return {
        "full_df": full_df,
        "train_df": train_df,
        "val_df": val_df,
        "file_summaries": file_summaries,
        "processed_files": processed_files,
        "skipped_files": skipped_files,
    }


def _drift_baseline_from_long_validation(
    full_df: pd.DataFrame, val_df: pd.DataFrame, h: int, window: int = 10
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build per-series drift baseline aligned to NHITS validation origin rows."""
    full_sorted = full_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    val_sorted = val_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    preds: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []

    for uid, g_val in val_sorted.groupby("unique_id", sort=False):
        g_val = g_val.sort_values("ds").reset_index(drop=True)
        if len(g_val) < h:
            continue
        first_val_ds = pd.to_datetime(g_val.loc[0, "ds"])
        g_full = full_sorted[full_sorted["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
        hist = g_full[pd.to_datetime(g_full["ds"]) < first_val_ds].reset_index(drop=True)
        if hist.empty:
            continue
        y_t = float(hist.iloc[-1]["y"])
        if len(hist) > window:
            y_prev = float(hist.iloc[-(window + 1)]["y"])
            slope = (y_t - y_prev) / float(window)
        else:
            slope = 0.0
        steps = np.arange(1, h + 1, dtype=float)
        preds.append(y_t + slope * steps)
        meta_rows.append(
            {"symbol": str(uid), "source_file": str(uid), "origin_ds": pd.to_datetime(hist.iloc[-1]["ds"])}
        )

    if not preds:
        return np.empty((0, h), dtype=float), pd.DataFrame(columns=["symbol", "source_file", "origin_ds"])
    return np.vstack(preds).astype(float, copy=False), pd.DataFrame(meta_rows)


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
        "persistence": _evaluate_with_corr_basis(y_val_np, pred_persistence, y0_val_np, use_delta_corr=delta_target),
        "drift": _evaluate_with_corr_basis(y_val_np, pred_drift, y0_val_np, use_delta_corr=delta_target),
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
    metrics["gbdt"] = _evaluate_with_corr_basis(y_val_np, pred_model, y0_val_np, use_delta_corr=delta_target)

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
        "models": models,
        "delta_target": bool(delta_target),
    }


def _run_lightgbm_experiment(
    ds: dict[str, object],
    *,
    horizon: int,
    delta_target: bool,
) -> dict[str, object]:
    """Train/evaluate baselines + LightGBM on a prebuilt tabular dataset."""
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
        "persistence": _evaluate_with_corr_basis(y_val_np, pred_persistence, y0_val_np, use_delta_corr=delta_target),
        "drift": _evaluate_with_corr_basis(y_val_np, pred_drift, y0_val_np, use_delta_corr=delta_target),
    }

    pred_model_delta = None
    try:
        if delta_target:
            train_target = Y_train.to_numpy(dtype=float) - y0_train_np.reshape(-1, 1)
            models = train_multioutput_lightgbm(X_train, train_target)
            pred_model_delta = predict_multioutput_lightgbm(models, X_val)
            pred_model = pred_model_delta + y0_val_np.reshape(-1, 1)
        else:
            models = train_multioutput_lightgbm(X_train, Y_train)
            pred_model = predict_multioutput_lightgbm(models, X_val)
    except ImportError as exc:
        raise SystemExit("LightGBM requested but lightgbm is not installed. Install `pip install lightgbm`.") from exc

    metrics["lightgbm"] = _evaluate_with_corr_basis(y_val_np, pred_model, y0_val_np, use_delta_corr=delta_target)
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
        "models": models,
        "delta_target": bool(delta_target),
    }


def _run_xgboost_experiment(
    ds: dict[str, object],
    *,
    horizon: int,
    delta_target: bool,
) -> dict[str, object]:
    """Train/evaluate baselines + XGBoost on a prebuilt tabular dataset."""
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
        "persistence": _evaluate_with_corr_basis(y_val_np, pred_persistence, y0_val_np, use_delta_corr=delta_target),
        "drift": _evaluate_with_corr_basis(y_val_np, pred_drift, y0_val_np, use_delta_corr=delta_target),
    }

    pred_model_delta = None
    try:
        if delta_target:
            train_target = Y_train.to_numpy(dtype=float) - y0_train_np.reshape(-1, 1)
            val_target = y_val_np - y0_val_np.reshape(-1, 1)
            models = train_multioutput_xgboost(X_train, train_target, X_val, val_target)
            pred_model_delta = predict_multioutput_xgboost(models, X_val)
            pred_model = pred_model_delta + y0_val_np.reshape(-1, 1)
        else:
            models = train_multioutput_xgboost(X_train, Y_train, X_val, Y_val)
            pred_model = predict_multioutput_xgboost(models, X_val)
    except ImportError as exc:
        raise SystemExit("XGBoost requested but xgboost is not installed. Install `pip install xgboost`.") from exc

    metrics["xgboost"] = _evaluate_with_corr_basis(y_val_np, pred_model, y0_val_np, use_delta_corr=delta_target)
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
        "models": models,
        "delta_target": bool(delta_target),
    }


def _horizon_band_mean(values: list[float], start_k: int, end_k: int) -> float:
    if not values:
        return 0.0
    start = max(1, start_k)
    end = min(len(values), end_k)
    if start > end:
        return 0.0
    arr = np.asarray(values[start - 1 : end], dtype=float)
    return float(arr.mean()) if arr.size else 0.0


def main() -> int:
    args = build_parser().parse_args()
    args.model = str(args.model).lower()
    if args.model not in ALL_MODELS:
        raise SystemExit(
            f"Unsupported --model '{args.model}'. Stable models: gbdt, dlinear, lightgbm, xgboost. "
            "Experimental: nhits, nbeats, tide, ensemble_gbdt_nhits (requires --experimental-neuralforecast)."
        )
    if args.model in EXPERIMENTAL_MODELS and not args.experimental_neuralforecast:
        raise SystemExit(
            f"Model '{args.model}' is experimental and disabled by default. "
            "Re-run with --experimental-neuralforecast to enable it."
        )
    if args.horizon < 1:
        raise SystemExit("--horizon must be >= 1")
    if args.lookback_lags < 1:
        raise SystemExit("--lookback-lags must be >= 1")
    if args.lookback < 1:
        raise SystemExit("--lookback must be >= 1")
    if args.input_size < 1:
        raise SystemExit("--input-size must be >= 1")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    if args.val_size < args.horizon:
        raise SystemExit("--val-size must be >= --horizon")
    if args.step_size < 1:
        raise SystemExit("--step-size must be >= 1")

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

    if args.model in {"gbdt", "lightgbm", "xgboost"}:
        ds = _assemble_tabular_dataset(args, selected_files)
        X_all = ds["X"]
        Y_all = ds["Y"]
        y0_all = ds["y0"]
        drift_all = ds["drift_pred"]
        meta_all = ds["meta"]
        n_rows = len(X_all)
    elif args.model == "dlinear":
        ds = _assemble_dlinear_dataset(args, selected_files)
        X_all = ds["X"]
        Y_all = ds["Y"]
        y0_all = ds["y0"]
        drift_all = ds["drift_pred"]
        meta_all = ds["meta"]
        n_rows = int(X_all.shape[0])
    elif args.model in {"nhits", "nbeats", "tide"}:
        ds = _assemble_nhits_dataset(args, selected_files)
        full_df = ds["full_df"]
        train_df = ds["train_df"]
        val_df = ds["val_df"]
        n_rows = int(len(full_df))
    else:
        # Ensemble path needs both tabular and NeuralForecast datasets.
        ds_tab = _assemble_tabular_dataset(args, selected_files)
        ds_nh = _assemble_nhits_dataset(args, selected_files)
        n_rows = int(len(ds_tab["X"]))

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
    elif args.model == "lightgbm":
        result = _run_lightgbm_experiment(ds, horizon=args.horizon, delta_target=bool(args.delta_target))
        pred_model = result["pred_model"]
        pred_model_delta = result["pred_model_delta"]
        pred_persistence = result["pred_persistence"]
        pred_drift = result["pred_drift"]
        y_val_np = result["y_val"]
        y0_val_np = result["y0_val"]
        split_idx = int(result["split_idx"])
        n_rows = int(result["n_rows"])
        metrics = result["metrics"]
        np.save(run_dir / "predictions_val.npy", pred_model)
        np.save(run_dir / "predictions_val_lightgbm.npy", pred_model)
        if pred_model_delta is not None:
            np.save(run_dir / "predictions_val_lightgbm_delta.npy", pred_model_delta)
        n_features = int(result["n_features"])
    elif args.model == "xgboost":
        result = _run_xgboost_experiment(ds, horizon=args.horizon, delta_target=bool(args.delta_target))
        pred_model = result["pred_model"]
        pred_model_delta = result["pred_model_delta"]
        pred_persistence = result["pred_persistence"]
        pred_drift = result["pred_drift"]
        y_val_np = result["y_val"]
        y0_val_np = result["y0_val"]
        split_idx = int(result["split_idx"])
        n_rows = int(result["n_rows"])
        metrics = result["metrics"]
        np.save(run_dir / "predictions_val.npy", pred_model)
        np.save(run_dir / "predictions_val_xgboost.npy", pred_model)
        if pred_model_delta is not None:
            np.save(run_dir / "predictions_val_xgboost_delta.npy", pred_model_delta)
        n_features = int(result["n_features"])
    elif args.model == "dlinear":
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
            "persistence": _evaluate_with_corr_basis(y_val_np, pred_persistence, y0_val_np, use_delta_corr=False),
            "drift": _evaluate_with_corr_basis(y_val_np, pred_drift, y0_val_np, use_delta_corr=False),
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
        metrics["dlinear"] = _evaluate_with_corr_basis(y_val_np, pred_model, y0_val_np, use_delta_corr=False)
        np.save(run_dir / "predictions_val.npy", pred_model)
        np.save(run_dir / "predictions_val_dlinear.npy", pred_model)
        n_features = int(X_train.shape[1])
    elif args.model in {"nhits", "nbeats", "tide"}:
        try:
            from src.models.neuralforecast_models import backtest_nf
        except ImportError as exc:
            raise SystemExit(
                f"{args.model.upper()} requested but neuralforecast is not installed. Install requirements.txt first."
            ) from exc
        full_df = ds["full_df"]
        try:
            y_val_np, y0_val_np, pred_model = backtest_nf(
                model_name=args.model,
                df_long=full_df,
                h=args.horizon,
                val_size=args.val_size,
                step_size=args.step_size,
                input_size=args.input_size,
                max_steps=args.max_steps,
                seed=args.seed,
            )
        except ImportError as exc:
            raise SystemExit(
                f"{args.model.upper()} requested but neuralforecast is not installed. Install requirements.txt first."
            ) from exc
        if y_val_np.shape[0] == 0:
            raise SystemExit(f"{args.model.upper()} backtest produced no aligned validation predictions.")
        n_val = int(y_val_np.shape[0])
        if n_val < 200 and not args.allow_small_val:
            print(
                f"WARNING: {args.model.upper()} backtest produced only {n_val} validation origins (<200). "
                "Rerun with larger --val-size or pass --allow-small-val to continue."
            )
            raise SystemExit(1)
        # For NF backtests with many rolling origins, use simple baselines aligned to the same y0/Y.
        pred_persistence = persistence_forecast(y0_val_np, horizon=args.horizon)
        pred_drift = drift_forecast(y0_val_np, horizon=args.horizon)
        meta_val = pd.DataFrame({"symbol": ["backtest"] * len(y0_val_np), "source_file": ["backtest"] * len(y0_val_np)})

        metrics = {
            "persistence": _evaluate_with_corr_basis(y_val_np, pred_persistence, y0_val_np, use_delta_corr=False),
            "drift": _evaluate_with_corr_basis(y_val_np, pred_drift, y0_val_np, use_delta_corr=False),
            args.model: _evaluate_with_corr_basis(y_val_np, pred_model, y0_val_np, use_delta_corr=False),
        }
        np.save(run_dir / "predictions_val.npy", pred_model)
        np.save(run_dir / f"predictions_val_{args.model}.npy", pred_model)
        split_idx = int(max(0, len(full_df) - args.val_size))
        n_rows = int(len(full_df))
        n_features = int(args.input_size)
    else:
        # Ensemble = 0.5 * GBDT(delta-target) + 0.5 * NHITS on absolute scale.
        try:
            from src.models.neuralforecast_models import predict_nf, train_nf
        except ImportError as exc:
            raise SystemExit(
                "ENSEMBLE_GBDT_NHITS requested but neuralforecast is not installed. Install requirements.txt first."
            ) from exc

        # Train GBDT with delta targets (forced) on tabular features.
        gbdt_result = _run_gbdt_experiment(ds_tab, horizon=args.horizon, delta_target=True)
        gbdt_models = gbdt_result["models"]
        assert isinstance(gbdt_models, list)

        # Train NHITS and get aligned validation arrays (one origin per series).
        full_df = ds_nh["full_df"]
        train_df = ds_nh["train_df"]
        val_df = ds_nh["val_df"]
        try:
            nf = train_nf(
                model_name="nhits",
                train_df=train_df,
                h=args.horizon,
                input_size=args.input_size,
                max_steps=args.max_steps,
                seed=args.seed,
            )
        except ImportError as exc:
            raise SystemExit(
                "ENSEMBLE_GBDT_NHITS requested but neuralforecast is not installed. Install requirements.txt first."
            ) from exc

        y_val_np, y0_val_np, pred_nhits = predict_nf(nf, full_df=full_df, val_df=val_df, h=args.horizon)
        if y_val_np.shape[0] == 0:
            raise SystemExit("NHITS produced no aligned validation predictions for ensemble.")
        pred_persistence = persistence_forecast(y0_val_np, horizon=args.horizon)
        pred_drift, meta_val = _drift_baseline_from_long_validation(full_df, val_df, h=args.horizon)
        if pred_drift.shape != y_val_np.shape:
            pred_drift = persistence_forecast(y0_val_np, horizon=args.horizon)
            meta_val = pd.DataFrame({"symbol": ["unknown"] * len(y0_val_np), "source_file": ["unknown"] * len(y0_val_np)})

        # Align GBDT predictions to the same NHITS validation origins using tabular metadata.
        tab_meta = ds_tab["meta"].copy().reset_index(drop=True)
        assert isinstance(tab_meta, pd.DataFrame)
        tab_meta["origin_ds"] = pd.to_datetime(tab_meta["origin_ds"])
        nh_meta = meta_val.copy().reset_index(drop=True)
        nh_meta["origin_ds"] = pd.to_datetime(nh_meta["origin_ds"])

        idx_pairs = list(zip(tab_meta["symbol"], tab_meta["origin_ds"]))
        pos_map = {(str(sym), pd.Timestamp(ds)): i for i, (sym, ds) in enumerate(idx_pairs)}
        match_positions: list[int] = []
        keep_rows: list[int] = []
        for i, row in nh_meta.iterrows():
            key = (str(row["symbol"]), pd.Timestamp(row["origin_ds"]))
            pos = pos_map.get(key)
            if pos is None:
                continue
            match_positions.append(pos)
            keep_rows.append(i)

        if not match_positions:
            raise SystemExit("Could not align GBDT tabular rows to NHITS validation origins for ensemble.")

        X_all_tab = ds_tab["X"]
        Y_all_tab = ds_tab["Y"]
        y0_all_tab = ds_tab["y0"]
        assert isinstance(X_all_tab, pd.DataFrame)
        assert isinstance(Y_all_tab, pd.DataFrame)
        assert isinstance(y0_all_tab, pd.Series)
        X_match = X_all_tab.iloc[match_positions]
        Y_match = Y_all_tab.iloc[match_positions].to_numpy(dtype=float)
        y0_match = y0_all_tab.iloc[match_positions].to_numpy(dtype=float)
        pred_gbdt_delta_match = predict_multioutput(gbdt_models, X_match)
        pred_gbdt_match = pred_gbdt_delta_match + y0_match.reshape(-1, 1)

        # Keep only rows that aligned in both models.
        y_val_np = y_val_np[keep_rows]
        y0_val_np = y0_val_np[keep_rows]
        pred_nhits = pred_nhits[keep_rows]
        pred_persistence = pred_persistence[keep_rows]
        pred_drift = pred_drift[keep_rows]
        meta_val = nh_meta.iloc[keep_rows].reset_index(drop=True)

        # Prefer the NHITS-aligned y/y0 for evaluation, but ensure shapes match.
        if pred_gbdt_match.shape != pred_nhits.shape or pred_gbdt_match.shape != y_val_np.shape:
            raise SystemExit("Aligned ensemble component prediction shapes do not match.")
        # If tabular targets differ slightly from NHITS alignment due to data cleaning, keep NHITS Y/y0.
        _ = Y_match, y0_match

        pred_model = average_ensemble(pred_gbdt_match, pred_nhits, weight_a=0.5)
        metrics = {
            "persistence": _evaluate_with_corr_basis(y_val_np, pred_persistence, y0_val_np, use_delta_corr=False),
            "drift": _evaluate_with_corr_basis(y_val_np, pred_drift, y0_val_np, use_delta_corr=False),
            "gbdt": _evaluate_with_corr_basis(y_val_np, pred_gbdt_match, y0_val_np, use_delta_corr=True),
            "nhits": _evaluate_with_corr_basis(y_val_np, pred_nhits, y0_val_np, use_delta_corr=False),
            "ensemble_gbdt_nhits": _evaluate_with_corr_basis(y_val_np, pred_model, y0_val_np, use_delta_corr=False),
        }
        np.save(run_dir / "predictions_val.npy", pred_model)
        np.save(run_dir / "predictions_val_gbdt.npy", pred_gbdt_match)
        np.save(run_dir / "predictions_val_nhits.npy", pred_nhits)
        np.save(run_dir / "predictions_val_ensemble_gbdt_nhits.npy", pred_model)
        split_idx = int(len(train_df))
        n_rows = int(len(train_df) + len(val_df))
        n_features = int(ds_tab["X"].shape[1])
        ds = {
            "processed_files": ds_nh["processed_files"],
            "skipped_files": ds_nh["skipped_files"],
            "file_summaries": ds_nh["file_summaries"],
        }

    np.save(run_dir / "predictions_val_persistence.npy", pred_persistence)
    np.save(run_dir / "predictions_val_drift.npy", pred_drift)
    np.save(run_dir / "y_val.npy", y_val_np)
    np.save(run_dir / "y0_val.npy", y0_val_np)

    if args.model not in {"nhits", "nbeats", "tide", "ensemble_gbdt_nhits"}:
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
        "val_fraction": (float(n_rows - split_idx) / float(n_rows)) if n_rows else None,
        "n_features": n_features,
        "file_summaries": ds["file_summaries"],
    }
    if args.model in {"gbdt", "lightgbm", "xgboost"}:
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
        trend_h = vals.get("trend_by_horizon", [])
        corr_h = vals.get("corr_by_horizon", [])
        t_near = _horizon_band_mean(trend_h, 1, 5)
        t_far = _horizon_band_mean(trend_h, 11, 15)
        c_near = _horizon_band_mean(corr_h, 1, 5)
        c_far = _horizon_band_mean(corr_h, 11, 15)
        print(
            f"  {name:12s} trend={vals['trend_accuracy']:.4f} "
            f"corr={vals['mean_correlation']:.4f} composite={vals['composite']:.4f}"
        )
        print(
            f"    horizons trend k1-5={t_near:.4f} k11-15={t_far:.4f} | "
            f"corr k1-5={c_near:.4f} k11-15={c_far:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
