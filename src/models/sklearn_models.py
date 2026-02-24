from __future__ import annotations

import numpy as np


def build_multioutput_gbr(random_state: int = 42):
    """Create a simple sklearn multi-output baseline model."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
    except ImportError as exc:
        raise ImportError("scikit-learn is required for model training") from exc

    base = HistGradientBoostingRegressor(random_state=random_state)
    return MultiOutputRegressor(base)


def train_multioutput_gbdt(X_train, Y_train):
    """Train one HistGradientBoostingRegressor per horizon column."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise ImportError("scikit-learn is required for model training") from exc

    X_arr = np.asarray(X_train)
    Y_arr = np.asarray(Y_train)
    if Y_arr.ndim != 2:
        raise ValueError("Y_train must be 2D with shape (n_samples, horizon)")
    if X_arr.shape[0] != Y_arr.shape[0]:
        raise ValueError("X_train and Y_train must have the same number of rows")

    models = []
    for h in range(Y_arr.shape[1]):
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42 + h,
        )
        model.fit(X_arr, Y_arr[:, h])
        models.append(model)
    return models


def predict_multioutput(models, X) -> np.ndarray:
    """Run per-horizon regressors and return shape (n_samples, horizon)."""
    X_arr = np.asarray(X)
    if len(models) == 0:
        return np.empty((X_arr.shape[0], 0), dtype=float)
    preds = [np.asarray(model.predict(X_arr), dtype=float) for model in models]
    return np.column_stack(preds)
