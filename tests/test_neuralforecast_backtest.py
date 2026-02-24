import os

import numpy as np
import pandas as pd
import pytest

from src.metrics.forecast import mean_horizon_correlation, trend_accuracy
from src.models.neuralforecast_models import _align_cv_to_arrays, backtest_nf, build_long_df


def test_align_cv_to_arrays_perfect_forecasts_on_increasing_series():
    n = 200
    h = 15
    val_size = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    y = np.arange(n, dtype=float)
    df_long = pd.DataFrame({"unique_id": "SYN", "ds": dates, "y": y})

    # Build perfect rolling forecasts for every cutoff in the validation tail that has h future steps.
    cutoffs = dates[n - val_size - 1 : n - h]
    rows = []
    for cutoff in cutoffs:
        cutoff_pos = dates.get_loc(cutoff)
        future_ds = dates[cutoff_pos + 1 : cutoff_pos + 1 + h]
        future_y = y[cutoff_pos + 1 : cutoff_pos + 1 + h]
        for ds, yy in zip(future_ds, future_y, strict=True):
            rows.append({"unique_id": "SYN", "cutoff": cutoff, "ds": ds, "NHITS": float(yy)})
    cv_df = pd.DataFrame(rows)

    Y_val, y0_val, y_pred = _align_cv_to_arrays(cv_df=cv_df, df_long=df_long, h=h, pred_col="NHITS")

    assert Y_val.shape[1] == h
    assert y_pred.shape == Y_val.shape
    assert y0_val.shape[0] == Y_val.shape[0]
    assert Y_val.shape[0] >= 30
    assert np.isfinite(Y_val).all()
    assert np.isfinite(y_pred).all()
    assert np.isfinite(y0_val).all()

    assert trend_accuracy(Y_val, y_pred, y0_val) == 1.0
    assert mean_horizon_correlation(Y_val, y_pred) >= 0.999999


def test_backtest_nf_nhits_shapes_smoke():
    if os.getenv("RUN_EXPERIMENTAL_TESTS") != "1":
        pytest.skip("Experimental NeuralForecast tests are disabled by default (set RUN_EXPERIMENTAL_TESTS=1).")
    pytest.importorskip("neuralforecast")

    n = 200
    h = 15
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    series = pd.Series(0.02 * t + np.sin(t / 8.0), index=dates)
    df_long = build_long_df([("SYN", series)])

    Y_val, y0_val, y_pred = backtest_nf(
        model_name="nhits",
        df_long=df_long,
        h=h,
        val_size=60,
        step_size=1,
        input_size=32,
        max_steps=5,
        seed=42,
    )

    assert Y_val.shape[1] == h
    assert y_pred.shape == Y_val.shape
    assert y0_val.shape[0] == Y_val.shape[0]
    assert Y_val.shape[0] >= 30
    assert np.isfinite(Y_val).all()
    assert np.isfinite(y_pred).all()
    assert np.isfinite(y0_val).all()
