from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _NFBundle:
    nf: object
    model_col: str


def build_long_df(per_file_series: list[tuple[str, pd.Series]]) -> pd.DataFrame:
    """Build NeuralForecast long-format dataframe from per-symbol series.

    Each series index must be datetime-like.
    """
    rows: list[pd.DataFrame] = []
    for symbol, series in per_file_series:
        if not isinstance(series, pd.Series):
            raise TypeError("Each item must be (symbol, pd.Series)")
        s = series.dropna().copy()
        if s.empty:
            continue
        ds = pd.to_datetime(s.index)
        if ds.isna().any():
            raise ValueError(f"Series for {symbol} has non-datetime index values")
        part = pd.DataFrame({"unique_id": str(symbol), "ds": ds, "y": s.to_numpy(dtype=float)})
        part = part.sort_values("ds").drop_duplicates(subset=["ds"], keep="last")
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])
    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return df[["unique_id", "ds", "y"]]


def split_long_df(df: pd.DataFrame, val_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split long df per series by time, reserving the last fraction for validation."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    required = {"unique_id", "ds", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"df must contain columns {required}")

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for _uid, g in df.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue
        n_val = max(1, int(np.ceil(n * val_fraction)))
        n_val = min(n - 1, n_val)
        split = n - n_val
        train_parts.append(g.iloc[:split].copy())
        val_parts.append(g.iloc[split:].copy())

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=df.columns)
    return train_df, val_df


def _infer_freq(train_df: pd.DataFrame) -> str:
    for _uid, g in train_df.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False):
        ds = pd.to_datetime(g["ds"]).sort_values()
        freq = pd.infer_freq(ds)
        if freq:
            return freq
        if len(ds) >= 2:
            delta = ds.iloc[1] - ds.iloc[0]
            if delta == pd.Timedelta(days=1):
                return "D"
            if delta == pd.Timedelta(hours=1):
                return "H"
            if delta == pd.Timedelta(minutes=5):
                return "5min"
    return "D"


def train_nf(model_name, train_df, h, input_size, max_steps, seed):
    """Train a NeuralForecast model (NHITS, NBEATS, or TiDE)."""
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS, NHITS, TiDE
    except ImportError as exc:
        raise ImportError("neuralforecast is required for train_nf") from exc

    model_name_l = str(model_name).lower()
    if model_name_l not in {"nhits", "nbeats", "tide"}:
        raise ValueError("model_name must be one of {'nhits','nbeats','tide'}")
    if h < 1 or input_size < 1 or max_steps < 1:
        raise ValueError("h, input_size, and max_steps must be >= 1")

    common = dict(h=int(h), input_size=int(input_size), max_steps=int(max_steps), random_seed=int(seed))
    # Optional args vary by version; pass only common args for compatibility.
    if model_name_l == "nhits":
        model = NHITS(**common)
    elif model_name_l == "nbeats":
        model = NBEATS(**common)
    else:
        model = TiDE(**common)

    freq = _infer_freq(train_df)
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=train_df.copy())

    # Forecast column is usually class name (e.g., 'NHITS').
    model_col = model.__class__.__name__
    return _NFBundle(nf=nf, model_col=model_col)


def _align_cv_to_arrays(cv_df: pd.DataFrame, df_long: pd.DataFrame, h: int, pred_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align NeuralForecast cross_validation output to (Y_val, y0_val, y_pred) arrays."""
    if h < 1:
        raise ValueError("h must be >= 1")
    required_cv = {"unique_id", "ds", "cutoff", pred_col}
    required_df = {"unique_id", "ds", "y"}
    if not required_cv.issubset(cv_df.columns):
        raise ValueError(f"cv_df must contain columns {required_cv}")
    if not required_df.issubset(df_long.columns):
        raise ValueError(f"df_long must contain columns {required_df}")

    full = df_long[["unique_id", "ds", "y"]].copy()
    full["unique_id"] = full["unique_id"].astype(str)
    full["ds"] = pd.to_datetime(full["ds"])
    full = full.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Index maps per series for exact horizon ordering checks.
    pos_maps: dict[str, dict[pd.Timestamp, int]] = {}
    ds_lists: dict[str, list[pd.Timestamp]] = {}
    y_maps: dict[str, dict[pd.Timestamp, float]] = {}
    for uid, g in full.groupby("unique_id", sort=False):
        ds_list = [pd.Timestamp(v) for v in pd.to_datetime(g["ds"]).tolist()]
        pos_maps[uid] = {ds: i for i, ds in enumerate(ds_list)}
        ds_lists[uid] = ds_list
        y_maps[uid] = {pd.Timestamp(ds): float(y) for ds, y in zip(ds_list, g["y"].to_numpy(dtype=float), strict=True)}

    cv = cv_df.copy()
    cv["unique_id"] = cv["unique_id"].astype(str)
    cv["ds"] = pd.to_datetime(cv["ds"])
    cv["cutoff"] = pd.to_datetime(cv["cutoff"])
    cv = cv.sort_values(["unique_id", "cutoff", "ds"]).reset_index(drop=True)

    y_true_rows: list[np.ndarray] = []
    y0_rows: list[float] = []
    y_pred_rows: list[np.ndarray] = []

    for (uid, cutoff), g in cv.groupby(["unique_id", "cutoff"], sort=False):
        if uid not in pos_maps:
            continue
        cutoff_ts = pd.Timestamp(cutoff)
        pos_map = pos_maps[uid]
        if cutoff_ts not in pos_map:
            continue
        cutoff_pos = pos_map[cutoff_ts]
        ds_list = ds_lists[uid]
        if cutoff_pos + h >= len(ds_list):
            continue
        expected_ds = ds_list[cutoff_pos + 1 : cutoff_pos + 1 + h]

        g = g.sort_values("ds")
        if len(g) != h:
            continue
        g_ds = [pd.Timestamp(v) for v in g["ds"].tolist()]
        if g_ds != expected_ds:
            continue

        y0 = y_maps[uid].get(cutoff_ts)
        if y0 is None:
            continue
        y_true = np.array([y_maps[uid][ds] for ds in expected_ds], dtype=float)
        y_pred = g[pred_col].to_numpy(dtype=float)

        if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all() and np.isfinite(y0)):
            continue
        y_true_rows.append(y_true)
        y_pred_rows.append(y_pred)
        y0_rows.append(float(y0))

    if not y_true_rows:
        return (
            np.empty((0, h), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0, h), dtype=float),
        )
    return (
        np.vstack(y_true_rows).astype(float, copy=False),
        np.asarray(y0_rows, dtype=float),
        np.vstack(y_pred_rows).astype(float, copy=False),
    )


def backtest_nf(
    model_name,
    df_long,
    h,
    *,
    val_size: int,
    step_size: int = 1,
    input_size: int,
    max_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NeuralForecast rolling backtest using cross_validation over the validation tail."""
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS, NHITS, TiDE
    except ImportError as exc:
        raise ImportError("neuralforecast is required for backtest_nf") from exc

    model_name_l = str(model_name).lower()
    if model_name_l not in {"nhits", "nbeats", "tide"}:
        raise ValueError("model_name must be one of {'nhits','nbeats','tide'}")
    if h < 1 or input_size < 1 or max_steps < 1 or val_size < h:
        raise ValueError("h/input_size/max_steps must be >=1 and val_size must be >= h")
    if step_size < 1:
        raise ValueError("step_size must be >= 1")

    df = df_long.copy()
    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    common = dict(h=int(h), input_size=int(input_size), max_steps=int(max_steps), random_seed=int(seed))
    if model_name_l == "nhits":
        model = NHITS(**common)
    elif model_name_l == "nbeats":
        model = NBEATS(**common)
    else:
        model = TiDE(**common)

    freq = _infer_freq(df)
    nf = NeuralForecast(models=[model], freq=freq)
    cv_df = nf.cross_validation(df=df, val_size=int(val_size), step_size=int(step_size), refit=False)
    # Some NeuralForecast versions return only one cutoff when refit=False + val_size is used.
    # Fallback to n_windows to force rolling origins while still deriving the extent from val_size.
    if "cutoff" in cv_df.columns and cv_df["cutoff"].nunique() <= 1 and val_size > h:
        n_windows = max(1, ((int(val_size) - int(h)) // int(step_size)) + 1)
        cv_df = nf.cross_validation(df=df, n_windows=n_windows, step_size=int(step_size), refit=False)

    pred_col = model.__class__.__name__
    if pred_col not in cv_df.columns:
        candidate_cols = [c for c in cv_df.columns if c not in {"unique_id", "ds", "cutoff", "y"}]
        if len(candidate_cols) != 1:
            raise ValueError(f"Could not determine forecast column in cross_validation output: {candidate_cols}")
        pred_col = candidate_cols[0]

    return _align_cv_to_arrays(cv_df=cv_df, df_long=df, h=int(h), pred_col=pred_col)


def predict_nf(nf, full_df, val_df, h):
    """Generate leakage-safe one-origin-per-series forecasts aligned to metric arrays.

    Uses a fitted NeuralForecast model trained on train_df (not provided here). We align against the
    first h points of each series' validation split and use the last pre-validation y as y0.
    Returns:
      Y_val: (n_series, h)
      y0_val: (n_series,)
      y_pred: (n_series, h)
    """
    if h < 1:
        raise ValueError("h must be >= 1")
    if not {"unique_id", "ds", "y"}.issubset(full_df.columns) or not {"unique_id", "ds", "y"}.issubset(val_df.columns):
        raise ValueError("full_df and val_df must contain unique_id, ds, y")

    if isinstance(nf, _NFBundle):
        bundle = nf
    else:
        # Backward-compatible: accept raw NeuralForecast object.
        model_col = None
        raw_nf = nf
        if hasattr(raw_nf, "models") and getattr(raw_nf, "models"):
            model_col = raw_nf.models[0].__class__.__name__
        if model_col is None:
            raise ValueError("Could not infer model column from provided NeuralForecast object")
        bundle = _NFBundle(nf=raw_nf, model_col=model_col)

    fcst = bundle.nf.predict()
    fcst = fcst.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    pred_col = bundle.model_col if bundle.model_col in fcst.columns else None
    if pred_col is None:
        candidate_cols = [c for c in fcst.columns if c not in {"unique_id", "ds"}]
        if len(candidate_cols) != 1:
            raise ValueError(f"Could not determine forecast column. Found: {candidate_cols}")
        pred_col = candidate_cols[0]

    full_sorted = full_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    val_sorted = val_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    y_true_rows: list[np.ndarray] = []
    y0_rows: list[float] = []
    y_pred_rows: list[np.ndarray] = []

    for uid, g_val in val_sorted.groupby("unique_id", sort=False):
        g_val = g_val.sort_values("ds").reset_index(drop=True)
        if len(g_val) < h:
            continue
        first_val_ds = pd.to_datetime(g_val.loc[0, "ds"])
        g_full = full_sorted[full_sorted["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
        hist = g_full[pd.to_datetime(g_full["ds"]) < first_val_ds]
        if hist.empty:
            continue
        y0 = float(hist.iloc[-1]["y"])
        y_true = g_val["y"].to_numpy(dtype=float)[:h]

        g_fcst = fcst[fcst["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
        if len(g_fcst) < h:
            continue
        y_pred = g_fcst[pred_col].to_numpy(dtype=float)[:h]

        if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all() and np.isfinite(y0)):
            continue
        y_true_rows.append(y_true)
        y_pred_rows.append(y_pred)
        y0_rows.append(y0)

    if not y_true_rows:
        return (
            np.empty((0, h), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0, h), dtype=float),
        )

    return (
        np.vstack(y_true_rows).astype(float, copy=False),
        np.asarray(y0_rows, dtype=float),
        np.vstack(y_pred_rows).astype(float, copy=False),
    )
