from __future__ import annotations

import numpy as np
import pandas as pd


def add_lag_features(df: pd.DataFrame, column: str, lags: list[int]) -> pd.DataFrame:
    """Add lag features using only past values."""
    out = df.copy()
    for lag in lags:
        if lag < 1:
            raise ValueError("Lag must be >= 1")
        out[f"{column}_lag_{lag}"] = out[column].shift(lag)
    return out


def add_rolling_mean_feature(df: pd.DataFrame, column: str, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """Add a leakage-safe rolling mean by shifting before rolling."""
    if window < 1:
        raise ValueError("window must be >= 1")
    out = df.copy()
    mp = min_periods if min_periods is not None else window
    out[f"{column}_rollmean_{window}"] = out[column].shift(1).rolling(window=window, min_periods=mp).mean()
    return out


def add_rolling_std_feature(df: pd.DataFrame, column: str, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """Add a leakage-safe rolling std by shifting before rolling."""
    if window < 1:
        raise ValueError("window must be >= 1")
    out = df.copy()
    mp = min_periods if min_periods is not None else window
    out[f"{column}_rollstd_{window}"] = out[column].shift(1).rolling(window=window, min_periods=mp).std()
    return out


def add_rolling_min_feature(df: pd.DataFrame, column: str, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """Add a leakage-safe rolling min by shifting before rolling."""
    if window < 1:
        raise ValueError("window must be >= 1")
    out = df.copy()
    mp = min_periods if min_periods is not None else window
    out[f"{column}_rollmin_{window}"] = out[column].shift(1).rolling(window=window, min_periods=mp).min()
    return out


def add_rolling_max_feature(df: pd.DataFrame, column: str, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """Add a leakage-safe rolling max by shifting before rolling."""
    if window < 1:
        raise ValueError("window must be >= 1")
    out = df.copy()
    mp = min_periods if min_periods is not None else window
    out[f"{column}_rollmax_{window}"] = out[column].shift(1).rolling(window=window, min_periods=mp).max()
    return out


def _rolling_slope(window_values: np.ndarray) -> float:
    if np.isnan(window_values).any():
        return np.nan
    n = window_values.shape[0]
    x = np.arange(n, dtype=float)
    x_mean = (n - 1) / 2.0
    y_mean = float(window_values.mean())
    denom = float(((x - x_mean) ** 2).sum())
    if denom == 0.0:
        return 0.0
    num = float(((x - x_mean) * (window_values - y_mean)).sum())
    return num / denom


def add_rolling_slope_feature(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    """Add a leakage-safe rolling linear-regression slope over past values."""
    if window < 1:
        raise ValueError("window must be >= 1")
    out = df.copy()
    shifted = out[column].shift(1)
    out[f"{column}_rollslope_{window}"] = shifted.rolling(window=window, min_periods=window).apply(
        _rolling_slope, raw=True
    )
    return out


def make_supervised_multi_horizon(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
    lags: list[int],
    roll_windows: list[int],
    include_stats: bool = True,
    return_origin_index: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Index]:
    """Build leakage-safe X/Y matrices for multi-horizon forecasting.

    Returns:
        X: feature matrix (lags + shifted rolling means only; no raw target at t)
        Y: multi-horizon targets with columns y_t+1..y_t+horizon
        y0: current target value at time t aligned to X/Y rows
    """
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if any(lag < 1 for lag in lags):
        raise ValueError("All lags must be >= 1")
    if any(window < 1 for window in roll_windows):
        raise ValueError("All roll_windows must be >= 1")

    target = df[target_col]

    feature_cols: dict[str, pd.Series] = {}
    for lag in lags:
        feature_cols[f"{target_col}_lag_{lag}"] = target.shift(lag)
    shifted = target.shift(1)
    for window in roll_windows:
        feature_cols[f"{target_col}_rollmean_{window}"] = shifted.rolling(window=window, min_periods=window).mean()
        if include_stats:
            feature_cols[f"{target_col}_rollstd_{window}"] = shifted.rolling(window=window, min_periods=window).std()
            feature_cols[f"{target_col}_rollmin_{window}"] = shifted.rolling(window=window, min_periods=window).min()
            feature_cols[f"{target_col}_rollmax_{window}"] = shifted.rolling(window=window, min_periods=window).max()
            feature_cols[f"{target_col}_rollslope_{window}"] = shifted.rolling(
                window=window, min_periods=window
            ).apply(_rolling_slope, raw=True)
    X = pd.DataFrame(feature_cols, index=df.index)

    y_cols = {f"y_t+{k}": target.shift(-k) for k in range(1, horizon + 1)}
    Y = pd.DataFrame(y_cols, index=df.index)
    y0 = target.copy()

    x_valid = X.notna().all(axis=1) if len(X.columns) > 0 else pd.Series(True, index=df.index)
    mask = x_valid & Y.notna().all(axis=1)

    origin_index = df.index[mask.to_numpy()]
    X = X.loc[mask].reset_index(drop=True)
    Y = Y.loc[mask].reset_index(drop=True)
    y0 = y0.loc[mask].reset_index(drop=True)
    if return_origin_index:
        return X, Y, y0, origin_index
    return X, Y, y0
