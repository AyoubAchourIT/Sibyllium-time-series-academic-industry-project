import numpy as np
import pytest

from src.metrics.forecast import composite_score
from src.models.sklearn_models import predict_multioutput, train_multioutput_gbdt


def test_gbdt_delta_target_pipeline_produces_finite_absolute_predictions(monkeypatch):
    sklearn = pytest.importorskip("sklearn")
    import sklearn.ensemble

    class DummyHGBR:
        def __init__(self, **kwargs):
            self.bias_ = 0.0

        def fit(self, X, y):
            y_arr = np.asarray(y, dtype=float)
            self.bias_ = float(np.mean(y_arr))
            return self

        def predict(self, X):
            X_arr = np.asarray(X)
            return np.full(X_arr.shape[0], self.bias_, dtype=float)

    monkeypatch.setattr(sklearn.ensemble, "HistGradientBoostingRegressor", DummyHGBR)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y0 = rng.normal(size=40)
    # Absolute multi-horizon targets with non-trivial deltas.
    Y = y0.reshape(-1, 1) + np.column_stack(
        [
            0.2 + 0.1 * rng.normal(size=40),
            0.4 + 0.1 * rng.normal(size=40),
            0.6 + 0.1 * rng.normal(size=40),
        ]
    )

    split = 30
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    y0_train, y0_val = y0[:split], y0[split:]

    D_train = Y_train - y0_train.reshape(-1, 1)
    models = train_multioutput_gbdt(X_train, D_train)
    pred_delta = predict_multioutput(models, X_val)
    pred_abs = pred_delta + y0_val.reshape(-1, 1)

    assert pred_delta.shape == Y_val.shape
    assert pred_abs.shape == Y_val.shape
    assert np.isfinite(pred_delta).all()
    assert np.isfinite(pred_abs).all()

    score = composite_score(Y_val, pred_abs, y0_val)
    assert np.isfinite(score)

