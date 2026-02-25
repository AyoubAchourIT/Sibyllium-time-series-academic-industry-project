from __future__ import annotations

import numpy as np


def train_multioutput_lightgbm(X_train, Y_train):
    """Train one LightGBM regressor per horizon column."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("lightgbm is required for LightGBM training") from exc

    X_arr = np.asarray(X_train)
    Y_arr = np.asarray(Y_train)
    if Y_arr.ndim != 2:
        raise ValueError("Y_train must be 2D with shape (n_samples, horizon)")
    if X_arr.shape[0] != Y_arr.shape[0]:
        raise ValueError("X_train and Y_train must have the same number of rows")

    models = []
    for h in range(Y_arr.shape[1]):
        model = LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42 + h,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(X_arr, Y_arr[:, h])
        models.append(model)
    return models


def predict_multioutput_lightgbm(models, X) -> np.ndarray:
    """Predict using per-horizon LightGBM regressors and stack to (n, horizon)."""
    X_arr = np.asarray(X)
    if len(models) == 0:
        return np.empty((X_arr.shape[0], 0), dtype=float)
    preds = [np.asarray(model.predict(X_arr), dtype=float) for model in models]
    return np.column_stack(preds)

