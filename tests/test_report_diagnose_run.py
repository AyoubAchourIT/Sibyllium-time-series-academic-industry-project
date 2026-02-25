import json

import numpy as np
import pandas as pd

from src.report.diagnose_run import diagnose_run


def test_diagnose_run_creates_expected_sections_on_fake_run(tmp_path):
    run_dir = tmp_path / "runs" / "RUN_FAKE"
    run_dir.mkdir(parents=True)

    n, h = 8, 4
    y0 = np.arange(10.0, 18.0)
    y_val = y0.reshape(-1, 1) + np.array([1.0, 2.0, 1.5, 3.0]).reshape(1, -1)
    y_pred = y_val.copy()

    np.save(run_dir / "y0_val.npy", y0)
    np.save(run_dir / "y_val.npy", y_val)
    np.save(run_dir / "predictions_val_gbdt.npy", y_pred)

    config = {"run_id": "RUN_FAKE", "target": "macd", "model": "gbdt", "delta_target": True, "limit_files": 2}
    metrics = {"gbdt": {"trend_accuracy": 1.0, "mean_correlation": 1.0, "composite": 1.0}}
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    preview = pd.DataFrame({"symbol": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC", "DDD", "DDD"]})
    preview.to_csv(run_dir / "preview_val.csv", index=False)

    out = diagnose_run(run_dir)
    out_path = run_dir / "diagnostics.json"
    out_path.write_text(json.dumps(out), encoding="utf-8")

    assert out_path.exists()
    assert "alignment_checks" in out
    assert "delta_stats" in out
    assert "horizon_metrics" in out
    assert "overall_metrics" in out
    assert out["alignment_checks"]["shapes"]["y_val"] == [n, h]
    assert len(out["horizon_metrics"]["trend_accuracy_k"]) == h
    assert len(out["horizon_metrics"]["corr_k"]) == h
    assert len(out["horizon_metrics"]["majority_baseline_trend"]) == h
    assert len(out["horizon_metrics"]["pct_small_delta"]) == h
