from __future__ import annotations

import numpy as np


def average_ensemble(pred_a, pred_b, weight_a: float = 0.5) -> np.ndarray:
    """Weighted average ensemble of two prediction matrices on absolute scale."""
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("pred_a and pred_b must have the same shape")
    if a.ndim != 2:
        raise ValueError("Predictions must be 2D arrays")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Predictions must be finite")
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError("weight_a must be in [0, 1]")
    out = weight_a * a + (1.0 - weight_a) * b
    return np.asarray(out, dtype=float)
