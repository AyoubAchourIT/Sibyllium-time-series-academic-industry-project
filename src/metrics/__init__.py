"""Forecast metrics for trend and correlation evaluation."""

from .forecast import (
    composite_score,
    correlation_by_horizon,
    mean_horizon_correlation,
    trend_accuracy,
    trend_accuracy_by_horizon,
)

__all__ = [
    "trend_accuracy",
    "trend_accuracy_by_horizon",
    "mean_horizon_correlation",
    "correlation_by_horizon",
    "composite_score",
]
