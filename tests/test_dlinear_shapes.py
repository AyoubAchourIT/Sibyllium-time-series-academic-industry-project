import numpy as np
import pandas as pd
import pytest

from src.features.windowing import make_windows_univariate


def test_dlinear_train_predict_shapes_tiny():
    torch = pytest.importorskip("torch")
    _ = torch  # keep linter quiet
    from src.models.dlinear import predict_dlinear, train_dlinear

    n = 180
    t = np.arange(n, dtype=float)
    series = pd.Series(0.05 * t + np.sin(t / 5.0))

    X, Y, y0 = make_windows_univariate(series, lookback=32, horizon=15)
    assert X.shape[1] == 32
    assert Y.shape[1] == 15
    assert len(y0) == len(X) == len(Y)

    split = max(1, min(len(X) - 1, int(len(X) * 0.8)))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    model = train_dlinear(X_train, Y_train, X_val, Y_val, epochs=1, batch_size=64, lr=1e-3)
    pred = predict_dlinear(model, X_val)

    assert pred.shape == (len(X_val), 15)
    assert np.isfinite(pred).all()

