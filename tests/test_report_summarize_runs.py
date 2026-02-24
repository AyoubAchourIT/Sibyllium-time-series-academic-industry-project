import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.report.summarize_runs import summarize_runs


def _write_run(
    run_dir: Path,
    *,
    model: str,
    target: str,
    composite: float,
    trend: float = 0.5,
    corr: float = 0.2,
    rows_total: int = 100,
    rows_val: int = 20,
    limit_files: int = 10,
    delta_target: bool = False,
    lookback_lags: int = 64,
    y_val_n: int = 20,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": model,
        "target": target,
        "rows_total": rows_total,
        "rows_val": rows_val,
        "limit_files": limit_files,
        "delta_target": delta_target,
        "lookback_lags": lookback_lags,
    }
    metrics = {
        "persistence": {"trend_accuracy": 0.0, "mean_correlation": 0.0, "composite": 0.0},
        "drift": {"trend_accuracy": 0.4, "mean_correlation": 0.1, "composite": 0.25},
        model: {"trend_accuracy": trend, "mean_correlation": corr, "composite": composite},
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    np.save(run_dir / "y_val.npy", np.zeros((y_val_n, 15), dtype=float))


def test_summarize_runs_sorts_by_composite_and_saves_expected_fields(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_run(runs_dir / "20260224T000001Z", model="gbdt", target="macd", composite=0.40, delta_target=True, y_val_n=42)
    _write_run(runs_dir / "20260224T000002Z", model="dlinear", target="stoch_k", composite=0.55, y_val_n=17)

    df = summarize_runs(runs_dir, last=20)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df["composite"]) == sorted(df["composite"], reverse=True)
    assert df.iloc[0]["run_id"] == "20260224T000002Z"
    assert df.iloc[0]["model"] == "dlinear"
    assert int(df.iloc[1]["n_val"]) == 42
    assert {"run_id", "target", "model", "trend", "corr", "composite", "val_fraction"}.issubset(df.columns)
    assert float(df.iloc[0]["val_fraction"]) == 0.2

