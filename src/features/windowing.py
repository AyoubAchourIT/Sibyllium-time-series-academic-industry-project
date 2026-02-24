from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows_univariate(series: pd.Series, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create leakage-safe univariate windows for direct multi-horizon forecasting.

    For each forecast origin t:
    - X[t] uses values [t-lookback+1, ..., t]
    - Y[t] uses values [t+1, ..., t+horizon]
    - y0[t] is the current value at origin t
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    values = series.to_numpy(dtype=float, copy=False)
    n = values.shape[0]
    if n < lookback + horizon:
        return (
            np.empty((0, lookback), dtype=float),
            np.empty((0, horizon), dtype=float),
            np.empty((0,), dtype=float),
        )

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    y0_rows: list[float] = []

    for t in range(lookback - 1, n - horizon):
        x = values[t - lookback + 1 : t + 1]
        y = values[t + 1 : t + 1 + horizon]
        y0 = values[t]
        if np.isnan(y0) or np.isnan(x).any() or np.isnan(y).any():
            continue
        x_rows.append(x.astype(float, copy=False))
        y_rows.append(y.astype(float, copy=False))
        y0_rows.append(float(y0))

    if not x_rows:
        return (
            np.empty((0, lookback), dtype=float),
            np.empty((0, horizon), dtype=float),
            np.empty((0,), dtype=float),
        )

    return (
        np.vstack(x_rows).astype(float, copy=False),
        np.vstack(y_rows).astype(float, copy=False),
        np.asarray(y0_rows, dtype=float),
    )

