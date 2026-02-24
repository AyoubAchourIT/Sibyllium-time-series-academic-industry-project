from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def persistence_forecast(y0: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat current value across forecast horizon.

    Args:
        y0: Current values with shape (n,) or (n, 1)
        horizon: Number of future steps
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    current = np.asarray(y0, dtype=float).reshape(-1, 1)
    if not np.isfinite(current).all():
        raise ValueError("y0 must contain only finite values")
    y_pred = np.repeat(current, horizon, axis=1).astype(float, copy=False)
    if y_pred.shape != (current.shape[0], horizon):
        raise ValueError("Unexpected persistence forecast shape")
    return y_pred


def drift_forecast(y: np.ndarray, horizon: int, window: int = 10) -> np.ndarray:
    """Linear drift forecast using a rolling slope estimate from past values.

    slope_t = (y_t - y_{t-window}) / window, with slope=0 when insufficient history.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if window < 1:
        raise ValueError("window must be >= 1")
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if not np.isfinite(y_arr).all():
        raise ValueError("y must contain only finite values")

    n = y_arr.shape[0]
    slopes = np.zeros(n, dtype=float)
    if n > window:
        slopes[window:] = (y_arr[window:] - y_arr[:-window]) / float(window)

    steps = np.arange(1, horizon + 1, dtype=float).reshape(1, -1)
    y_pred = y_arr.reshape(-1, 1) + slopes.reshape(-1, 1) * steps
    return y_pred.astype(float, copy=False)


@dataclass(slots=True)
class PersistenceBaseline:
    horizon: int = 15

    def predict(self, current_values):
        return persistence_forecast(current_values, self.horizon)


@dataclass(slots=True)
class DriftBaseline:
    horizon: int = 15

    def predict(self, current_values, previous_values):
        current = np.asarray(current_values, dtype=float).reshape(-1, 1)
        previous = np.asarray(previous_values, dtype=float).reshape(-1, 1)
        slope = current - previous
        steps = np.arange(1, self.horizon + 1, dtype=float).reshape(1, -1)
        return current + slope * steps
