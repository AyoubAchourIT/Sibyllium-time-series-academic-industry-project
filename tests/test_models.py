import numpy as np
import pytest

from src.models.baselines import persistence_forecast
from src.models.sklearn_models import predict_multioutput, train_multioutput_gbdt


def test_persistence_forecast_shape_and_values():
    y0 = np.array([1.5, -2.0, 3.0])
    out = persistence_forecast(y0, horizon=4)
    assert out.shape == (3, 4)
    assert np.allclose(out[0], [1.5, 1.5, 1.5, 1.5])
    assert np.allclose(out[1], [-2.0, -2.0, -2.0, -2.0])


def test_train_and_predict_multioutput_gbdt_shapes(monkeypatch):
    sklearn = pytest.importorskip("sklearn")
    import sklearn.ensemble

    class DummyHGBR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.bias_ = 0.0

        def fit(self, X, y):
            X_arr = np.asarray(X)
            y_arr = np.asarray(y)
            assert X_arr.ndim == 2
            self.bias_ = float(np.mean(y_arr))
            return self

        def predict(self, X):
            X_arr = np.asarray(X)
            return np.full(X_arr.shape[0], self.bias_, dtype=float)

    monkeypatch.setattr(sklearn.ensemble, "HistGradientBoostingRegressor", DummyHGBR)

    rng = np.random.default_rng(42)
    n = 80
    X = rng.normal(size=(n, 4))
    y_base = X[:, 0] * 0.5 + X[:, 1] * -0.2
    Y = np.column_stack(
        [
            y_base + 0.1,
            y_base * 0.8 + 0.2,
            y_base * -0.1 + 0.3,
        ]
    )

    models = train_multioutput_gbdt(X, Y)
    preds = predict_multioutput(models, X[:10])

    assert isinstance(models, list)
    assert len(models) == 3
    assert preds.shape == (10, 3)
    assert np.isfinite(preds).all()
