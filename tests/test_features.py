import pandas as pd

from src.features.tabular import add_rolling_mean_feature, make_supervised_multi_horizon


def test_rolling_feature_is_shifted_to_prevent_leakage():
    df = pd.DataFrame({"close": [10, 20, 30, 40]})
    out = add_rolling_mean_feature(df, "close", window=2, min_periods=2)
    values = out["close_rollmean_2"].tolist()
    assert pd.isna(values[0])
    assert pd.isna(values[1])
    assert values[2] == 15  # mean(10,20), does not include current row (30)
    assert values[3] == 25  # mean(20,30), does not include current row (40)


def test_make_supervised_multi_horizon_builds_aligned_leakage_safe_xy():
    df = pd.DataFrame({"close": list(range(10))})  # increasing synthetic series

    X, Y, y0 = make_supervised_multi_horizon(
        df=df,
        target_col="close",
        horizon=3,
        lags=[1, 2],
        roll_windows=[2],
        include_stats=False,
    )

    assert X.shape == (5, 3)
    assert Y.shape == (5, 3)
    assert y0.shape == (5,)
    assert len(X) == len(Y) == len(y0)

    assert not X.isna().any().any()
    assert not Y.isna().any().any()
    assert not y0.isna().any()

    assert list(Y.columns) == ["y_t+1", "y_t+2", "y_t+3"]
    assert Y.iloc[0].tolist() == [3, 4, 5]
    assert Y.iloc[-1].tolist() == [7, 8, 9]
    assert y0.iloc[0] == 2
    assert y0.iloc[-1] == 6

    assert "close" not in X.columns
    assert set(X.columns) == {"close_lag_1", "close_lag_2", "close_rollmean_2"}


def test_make_supervised_multi_horizon_includes_richer_rolling_stats_when_enabled():
    df = pd.DataFrame({"close": [float(i) for i in range(20)]})

    X, Y, y0 = make_supervised_multi_horizon(
        df=df,
        target_col="close",
        horizon=4,
        lags=[1, 2, 3],
        roll_windows=[3],
        include_stats=True,
    )

    expected_cols = {
        "close_lag_1",
        "close_lag_2",
        "close_lag_3",
        "close_rollmean_3",
        "close_rollstd_3",
        "close_rollmin_3",
        "close_rollmax_3",
        "close_rollslope_3",
    }

    assert expected_cols.issubset(set(X.columns))
    assert "close" not in X.columns
    assert len(X) == len(Y) == len(y0)
    assert not X.isna().any().any()
    assert not Y.isna().any().any()
    assert not y0.isna().any()
