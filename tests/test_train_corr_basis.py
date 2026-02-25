import numpy as np

from src.train import _evaluate_with_corr_basis


def test_evaluate_with_corr_basis_delta_selects_delta_corr_series():
    y0 = np.array([10.0, 20.0, 30.0])
    y_true = np.array(
        [
            [11.0, 12.0, 13.0],
            [21.0, 21.5, 22.5],
            [29.5, 29.0, 28.5],
        ]
    )
    # Predictions correlated on deltas but not identical on levels.
    y_pred = np.array(
        [
            [10.8, 11.7, 12.8],
            [20.9, 21.3, 22.2],
            [29.7, 29.2, 28.8],
        ]
    )

    out = _evaluate_with_corr_basis(y_true, y_pred, y0, use_delta_corr=True)

    assert out["corr_basis"] == "delta"
    assert np.allclose(out["corr_by_horizon"], out["corr_delta_by_horizon"])
    assert np.isclose(out["mean_correlation"], np.mean(out["corr_delta_by_horizon"]))


def test_evaluate_with_corr_basis_level_selects_level_corr_series():
    y0 = np.array([1.0, 2.0, 3.0, 4.0])
    y_true = np.array(
        [
            [1.1, 1.2],
            [2.2, 2.3],
            [3.3, 3.1],
            [4.4, 4.2],
        ]
    )
    y_pred = np.array(
        [
            [1.0, 1.25],
            [2.1, 2.35],
            [3.2, 3.05],
            [4.6, 4.10],
        ]
    )

    out = _evaluate_with_corr_basis(y_true, y_pred, y0, use_delta_corr=False)

    assert out["corr_basis"] == "level"
    assert np.allclose(out["corr_by_horizon"], out["corr_level_by_horizon"])
    assert np.isclose(out["mean_correlation"], np.mean(out["corr_level_by_horizon"]))

