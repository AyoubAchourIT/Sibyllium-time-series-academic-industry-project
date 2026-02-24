from __future__ import annotations

import numpy as np


ArrayLike = np.ndarray


def _as_2d(a: ArrayLike) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Expected 1D or 2D array")
    return arr


def trend_accuracy(y_true: ArrayLike, y_pred: ArrayLike, y_current: ArrayLike | float) -> float:
    """Sign agreement of horizon deltas, averaged over samples and horizons."""
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    yc = np.asarray(y_current, dtype=float)
    if yc.ndim == 0:
        yc = np.full((yt.shape[0], 1), yc)
    elif yc.ndim == 1:
        yc = yc.reshape(-1, 1)
    if yt.shape != yp.shape or yc.shape[0] != yt.shape[0]:
        raise ValueError("Shape mismatch between inputs")
    if not (np.isfinite(yt).all() and np.isfinite(yp).all() and np.isfinite(yc).all()):
        raise ValueError("Inputs must be finite")
    true_sign = np.sign(yt - yc)
    pred_sign = np.sign(yp - yc)
    score = float((true_sign == pred_sign).mean())
    if not np.isfinite(score):
        raise ValueError("trend_accuracy produced a non-finite result")
    return score


def trend_accuracy_by_horizon(y_true: ArrayLike, y_pred: ArrayLike, y_current: ArrayLike | float) -> list[float]:
    """Per-horizon sign agreement averaged over samples."""
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    yc = np.asarray(y_current, dtype=float)
    if yc.ndim == 0:
        yc = np.full((yt.shape[0], 1), yc)
    elif yc.ndim == 1:
        yc = yc.reshape(-1, 1)
    if yt.shape != yp.shape or yc.shape[0] != yt.shape[0]:
        raise ValueError("Shape mismatch between inputs")
    if not (np.isfinite(yt).all() and np.isfinite(yp).all() and np.isfinite(yc).all()):
        raise ValueError("Inputs must be finite")
    true_sign = np.sign(yt - yc)
    pred_sign = np.sign(yp - yc)
    scores = (true_sign == pred_sign).mean(axis=0)
    scores = np.asarray(scores, dtype=float)
    scores[~np.isfinite(scores)] = 0.0
    return scores.tolist()


def mean_horizon_correlation(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Average per-sample Pearson correlation over horizon vectors."""
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    corrs: list[float] = []
    for row_true, row_pred in zip(yt, yp, strict=True):
        if np.std(row_true) == 0 or np.std(row_pred) == 0:
            corrs.append(0.0)
            continue
        corr = float(np.corrcoef(row_true, row_pred)[0, 1])
        corrs.append(corr if np.isfinite(corr) else 0.0)
    return float(np.mean(corrs)) if corrs else 0.0


def correlation_by_horizon(y_true: ArrayLike, y_pred: ArrayLike) -> list[float]:
    """Per-horizon Pearson correlation across samples."""
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    out: list[float] = []
    for k in range(yt.shape[1]):
        col_true = yt[:, k]
        col_pred = yp[:, k]
        if np.std(col_true) == 0 or np.std(col_pred) == 0:
            out.append(0.0)
            continue
        corr = float(np.corrcoef(col_true, col_pred)[0, 1])
        out.append(corr if np.isfinite(corr) else 0.0)
    return out


def composite_score(y_true: ArrayLike, y_pred: ArrayLike, y_current: ArrayLike | float, alpha: float = 0.5) -> float:
    """Blend trend and correlation with weight alpha for trend."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    trend = trend_accuracy(y_true, y_pred, y_current)
    corr = mean_horizon_correlation(y_true, y_pred)
    return float(alpha * trend + (1 - alpha) * corr)
