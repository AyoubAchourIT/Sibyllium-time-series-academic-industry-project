"""Forecast metrics for trend and correlation evaluation."""

from .forecast import composite_score, mean_horizon_correlation, trend_accuracy

__all__ = ["trend_accuracy", "mean_horizon_correlation", "composite_score"]
