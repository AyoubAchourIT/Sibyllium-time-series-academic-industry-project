from __future__ import annotations

import pandas as pd


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD columns from a DataFrame containing `close`."""
    ema_fast = _ema(df["close"], fast)
    ema_slow = _ema(df["close"], slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    return pd.DataFrame(
        {
            f"ema_{fast}": ema_fast,
            f"ema_{slow}": ema_slow,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_histogram": macd - macd_signal,
        }
    )


def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_period: int = 5,
) -> pd.DataFrame:
    """Compute stochastic oscillator columns from `high`, `low`, `close`."""
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    denom = (high_max - low_min).replace(0, pd.NA)
    stoch_k = 100 * (df["close"] - low_min) / denom
    stoch_d = stoch_k.rolling(window=d_period).mean()
    stoch_d_smooth = stoch_d.rolling(window=smooth_period).mean()
    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d, "stoch_d_smooth": stoch_d_smooth})
