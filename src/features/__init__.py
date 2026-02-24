"""Feature engineering helpers with leakage-safe shifting."""

from .tabular import (
    add_lag_features,
    add_rolling_max_feature,
    add_rolling_mean_feature,
    add_rolling_min_feature,
    add_rolling_slope_feature,
    add_rolling_std_feature,
    make_supervised_multi_horizon,
)
from .technical import calculate_macd, calculate_stochastic
from .windowing import make_windows_univariate

__all__ = [
    "add_lag_features",
    "add_rolling_mean_feature",
    "add_rolling_std_feature",
    "add_rolling_min_feature",
    "add_rolling_max_feature",
    "add_rolling_slope_feature",
    "make_supervised_multi_horizon",
    "make_windows_univariate",
    "calculate_macd",
    "calculate_stochastic",
]
