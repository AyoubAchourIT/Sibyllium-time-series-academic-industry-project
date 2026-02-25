from __future__ import annotations

import numpy as np


def train_multioutput_xgboost(X_train, Y_train, X_val=None, Y_val=None):
    """Train one XGBoost regressor per horizon column."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is required for XGBoost training") from exc

    X_train_arr = np.asarray(X_train)
    Y_train_arr = np.asarray(Y_train)
    if Y_train_arr.ndim != 2:
        raise ValueError("Y_train must be 2D with shape (n_samples, horizon)")
    if X_train_arr.shape[0] != Y_train_arr.shape[0]:
        raise ValueError("X_train and Y_train must have the same number of rows")

    X_val_arr = None if X_val is None else np.asarray(X_val)
    Y_val_arr = None if Y_val is None else np.asarray(Y_val)
    if X_val_arr is not None and Y_val_arr is not None:
        if Y_val_arr.ndim != 2:
            raise ValueError("Y_val must be 2D with shape (n_samples, horizon)")
        if X_val_arr.shape[0] != Y_val_arr.shape[0]:
            raise ValueError("X_val and Y_val must have the same number of rows")
        if Y_val_arr.shape[1] != Y_train_arr.shape[1]:
            raise ValueError("Y_val horizon must match Y_train horizon")

    models = []
    for h in range(Y_train_arr.shape[1]):
        model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            eval_metric="rmse",
        )
        fit_kwargs = {}
        if X_val_arr is not None and Y_val_arr is not None and X_val_arr.shape[0] > 0:
            fit_kwargs = {
                "eval_set": [(X_val_arr, Y_val_arr[:, h])],
                "verbose": False,
            }
        try:
            model.fit(X_train_arr, Y_train_arr[:, h], **fit_kwargs)
        except TypeError:
            # Older/newer xgboost sklearn APIs vary. Retry without optional fit kwargs.
            model.fit(X_train_arr, Y_train_arr[:, h])
        models.append(model)
    return models


def predict_multioutput_xgboost(models, X) -> np.ndarray:
    """Predict using per-horizon XGBoost regressors and stack to (n, horizon)."""
    X_arr = np.asarray(X)
    if len(models) == 0:
        return np.empty((X_arr.shape[0], 0), dtype=float)
    preds = [np.asarray(model.predict(X_arr), dtype=float) for model in models]
    return np.column_stack(preds)
