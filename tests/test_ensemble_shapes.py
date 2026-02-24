import numpy as np

from src.models.ensembles import average_ensemble


def test_average_ensemble_shape_and_finiteness():
    a = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]], dtype=float)
    b = np.array([[1.5, 1.0, 2.0], [1.5, 1.0, 6.0]], dtype=float)

    out = average_ensemble(a, b, weight_a=0.5)

    assert out.shape == a.shape
    assert np.isfinite(out).all()
    assert np.allclose(out, 0.5 * a + 0.5 * b)

