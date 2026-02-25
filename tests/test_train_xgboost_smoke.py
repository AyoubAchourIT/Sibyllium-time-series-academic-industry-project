import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_train_xgboost_writes_standard_artifacts(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")

    import src.train as train_mod

    n, h = 60, 3
    X = pd.DataFrame(
        {
            "x1": np.linspace(0, 1, n),
            "x2": np.sin(np.linspace(0, 4, n)),
            "x3": np.cos(np.linspace(0, 3, n)),
        }
    )
    y0 = pd.Series(np.linspace(10, 20, n))
    Y = pd.DataFrame({"y_t+1": y0 + 0.1, "y_t+2": y0 + 0.2, "y_t+3": y0 + 0.3})
    drift = np.repeat(y0.to_numpy().reshape(-1, 1), h, axis=1)
    meta = pd.DataFrame(
        {
            "symbol": ["SYN"] * n,
            "source_file": ["SYN.xlsx"] * n,
            "origin_ds": pd.date_range("2024-01-01", periods=n),
        }
    )
    ds = {
        "X": X,
        "Y": Y,
        "y0": y0,
        "drift_pred": drift,
        "meta": meta,
        "lags": [1, 2, 3],
        "roll_windows": [3, 5],
        "processed_files": 1,
        "skipped_files": 0,
        "file_summaries": [{"symbol": "SYN", "file": "SYN.xlsx", "rows_supervised": n}],
    }

    monkeypatch.setattr(train_mod, "_select_files", lambda *args, **kwargs: [Path("SYN.xlsx")])
    monkeypatch.setattr(train_mod, "_assemble_tabular_dataset", lambda *args, **kwargs: ds)
    monkeypatch.setattr(train_mod, "train_multioutput_xgboost", lambda Xtr, Ytr, Xv=None, Yv=None: [object()] * Ytr.shape[1])
    monkeypatch.setattr(
        train_mod,
        "predict_multioutput_xgboost",
        lambda models, Xv: np.tile(np.asarray(y0.iloc[-len(Xv):], dtype=float).reshape(-1, 1), (1, len(models))),
    )

    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "src.train",
            "--runs-dir",
            str(runs_dir),
            "--data-dir",
            "datas",
            "--model",
            "xgboost",
            "--target",
            "stoch_k",
            "--horizon",
            "3",
            "--delta-target",
            "--lookback-lags",
            "8",
        ],
    )

    rc = train_mod.main()
    assert rc == 0

    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    pred_path = run_dir / "predictions_val_xgboost.npy"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    assert pred_path.exists()
    assert metrics_path.exists()
    assert config_path.exists()

    preds = np.load(pred_path)
    assert preds.shape == (12, h)
    assert np.isfinite(preds).all()

    metrics = json.loads(metrics_path.read_text())
    config = json.loads(config_path.read_text())
    assert "xgboost" in metrics
    assert metrics["xgboost"]["corr_basis"] == "delta"
    assert "corr_by_horizon" in metrics["xgboost"]
    assert config["model"] == "xgboost"
