import numpy as np

from src.metrics.forecast import composite_score, mean_horizon_correlation, trend_accuracy
from src.models.baselines import drift_forecast, persistence_forecast


def test_metrics_perfect_prediction_scores_one():
    y_current = np.array([1.0])
    y_true = np.array([[1.1, 1.2, 1.3]])
    y_pred = y_true.copy()
    assert trend_accuracy(y_true, y_pred, y_current) == 1.0
    assert mean_horizon_correlation(y_true, y_pred) == 1.0
    assert composite_score(y_true, y_pred, y_current) == 1.0


def test_trend_accuracy_penalizes_wrong_direction():
    y_current = np.array([1.0])
    y_true = np.array([[1.1, 0.9, 1.2]])
    y_pred = np.array([[0.9, 1.1, 0.8]])
    assert trend_accuracy(y_true, y_pred, y_current) == 0.0


def test_persistence_baseline_and_metrics_are_finite_on_increasing_targets():
    y0 = np.array([1.0, 2.0, 3.0], dtype=float)
    # Strictly increasing future trajectories from current value.
    Y = np.array(
        [
            [1.1, 1.2, 1.3, 1.4],
            [2.2, 2.3, 2.4, 2.5],
            [3.5, 3.6, 3.7, 3.8],
        ],
        dtype=float,
    )

    pred_persistence = persistence_forecast(y0, horizon=Y.shape[1])

    assert pred_persistence.shape == Y.shape
    assert np.allclose(pred_persistence, y0.reshape(-1, 1))
    assert np.isfinite(pred_persistence).all()

    trend_p = trend_accuracy(Y, pred_persistence, y0)
    corr_p = mean_horizon_correlation(Y, pred_persistence)
    assert 0.0 <= trend_p <= 1.0
    assert not np.isnan(trend_p)
    assert not np.isnan(corr_p)

    trend_perfect = trend_accuracy(Y, Y, y0)
    assert trend_perfect >= 0.99


def test_drift_baseline_beats_persistence_on_increasing_series_trend():
    y0 = np.arange(10.0, 22.0, 1.0)  # 12 samples, strictly increasing current values
    horizon = 4
    # Future paths remain increasing for every sample.
    Y = y0.reshape(-1, 1) + np.arange(1, horizon + 1, dtype=float).reshape(1, -1)

    pred_persistence = persistence_forecast(y0, horizon=horizon)
    pred_drift = drift_forecast(y0, horizon=horizon, window=2)

    trend_p = trend_accuracy(Y, pred_persistence, y0)
    trend_d = trend_accuracy(Y, pred_drift, y0)

    assert pred_drift.shape == Y.shape
    assert np.isfinite(pred_drift).all()
    assert trend_d > trend_p
