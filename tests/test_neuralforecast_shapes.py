import os

import numpy as np
import pandas as pd
import pytest

from src.models.neuralforecast_models import build_long_df, predict_nf, split_long_df, train_nf


def test_neuralforecast_nhits_shapes_tiny():
    if os.getenv("RUN_EXPERIMENTAL_TESTS") != "1":
        pytest.skip("Experimental NeuralForecast tests are disabled by default (set RUN_EXPERIMENTAL_TESTS=1).")
    pytest.importorskip("neuralforecast")

    n = 120
    h = 12
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    series = pd.Series(0.03 * t + np.sin(t / 6.0), index=dates)

    full_df = build_long_df([("SYN", series)])
    train_df, val_df = split_long_df(full_df, val_fraction=0.25)
    assert not train_df.empty and not val_df.empty

    nf = train_nf(
        model_name="nhits",
        train_df=train_df,
        h=h,
        input_size=24,
        max_steps=5,
        seed=42,
    )
    Y_val, y0_val, y_pred = predict_nf(nf, full_df, val_df, h=h)

    assert Y_val.shape == (1, h)
    assert y_pred.shape == (1, h)
    assert y0_val.shape == (1,)
    assert np.isfinite(Y_val).all()
    assert np.isfinite(y_pred).all()
    assert np.isfinite(y0_val).all()
