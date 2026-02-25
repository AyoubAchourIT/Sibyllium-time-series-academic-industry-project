from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics.forecast import composite_score, mean_horizon_correlation, trend_accuracy


PRED_CANDIDATES = [
    "predictions_val_gbdt.npy",
    "predictions_val_lightgbm.npy",
    "predictions_val_xgboost.npy",
    "predictions_val_dlinear.npy",
    "predictions_val_nhits.npy",
    "predictions_val_nbeats.npy",
    "predictions_val_tide.npy",
    "predictions_val_ensemble_gbdt_nhits.npy",
    "predictions_val.npy",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnose a run directory (alignment, horizon metrics, symbol breakdown).")
    p.add_argument("--run-dir", required=True, help="Path to runs/<RUN_ID>")
    p.add_argument("--out", default=None, help="Output JSON path (default: <run-dir>/diagnostics.json)")
    return p


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_pred(run_dir: Path, config: dict[str, Any]) -> tuple[np.ndarray, str]:
    model = str(config.get("model", "")).lower()
    if model:
        preferred = run_dir / f"predictions_val_{model}.npy"
        if preferred.exists():
            return np.load(preferred), preferred.name
    for name in PRED_CANDIDATES:
        path = run_dir / name
        if path.exists():
            return np.load(path), name
    raise SystemExit(f"No prediction file found in {run_dir}")


def _finite_stats(arr: np.ndarray) -> dict[str, Any]:
    total = int(arr.size)
    non_finite = int(np.size(arr) - np.isfinite(arr).sum())
    return {
        "total_values": total,
        "non_finite_count": non_finite,
        "non_finite_pct": (100.0 * non_finite / total) if total else 0.0,
    }


def _corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else 0.0


def _trend_by_horizon(delta_true: np.ndarray, delta_pred: np.ndarray) -> list[float]:
    return [float(np.mean(np.sign(delta_true[:, k]) == np.sign(delta_pred[:, k]))) for k in range(delta_true.shape[1])]


def _corr_by_horizon(y_true: np.ndarray, y_pred: np.ndarray, use_delta: bool, y0: np.ndarray) -> list[float]:
    if use_delta:
        a = y_true - y0.reshape(-1, 1)
        b = y_pred - y0.reshape(-1, 1)
    else:
        a = y_true
        b = y_pred
    return [_corr_1d(a[:, k], b[:, k]) for k in range(a.shape[1])]


def _band_indices(h: int) -> dict[str, list[int]]:
    return {
        "k1_5": [k for k in range(h) if 0 <= k < min(5, h)],
        "k6_10": [k for k in range(h) if 5 <= k < min(10, h)],
        "k11_15": [k for k in range(h) if 10 <= k < min(15, h)],
    }


def _delta_stats(delta_true: np.ndarray) -> dict[str, Any]:
    h = delta_true.shape[1]
    out: dict[str, Any] = {"by_horizon": {}, "buckets": {}}
    for k in range(h):
        d = delta_true[:, k]
        p = float(np.mean(d > 0)) if d.size else 0.0
        out["by_horizon"][f"k{k+1}"] = {
            "mean_abs_delta": float(np.mean(np.abs(d))) if d.size else 0.0,
            "std_delta": float(np.std(d)) if d.size else 0.0,
            "positive_rate": p,
            "balance_p1mp": float(p * (1.0 - p)),
        }
    bands = _band_indices(h)
    for name, idxs in bands.items():
        if not idxs:
            out["buckets"][name] = {}
            continue
        d = delta_true[:, idxs].reshape(-1)
        p = float(np.mean(d > 0)) if d.size else 0.0
        out["buckets"][name] = {
            "mean_abs_delta": float(np.mean(np.abs(d))) if d.size else 0.0,
            "std_delta": float(np.std(d)) if d.size else 0.0,
            "positive_rate": p,
            "balance_p1mp": float(p * (1.0 - p)),
            "n_values": int(d.size),
        }
    return out


def _compute_symbol_breakdown(preview_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y0: np.ndarray) -> dict[str, Any]:
    if "symbol" not in preview_df.columns:
        return {"available": False, "reason": "preview_val.csv has no symbol column"}
    n = min(len(preview_df), y_true.shape[0])
    if n == 0:
        return {"available": False, "reason": "empty preview or arrays"}
    if n < y_true.shape[0]:
        note = f"preview subset used ({n}/{y_true.shape[0]} rows)"
    else:
        note = "full validation rows covered"

    p = preview_df.iloc[:n].copy().reset_index(drop=True)
    rows = []
    trend_h = _trend_by_horizon(y_true[:n] - y0[:n, None], y_pred[:n] - y0[:n, None])  # global fallback shape check
    _ = trend_h
    for sym, idx in p.groupby("symbol", sort=False).groups.items():
        idx = np.array(list(idx), dtype=int)
        if idx.size < 2:
            continue
        yt = y_true[idx]
        yp = y_pred[idx]
        y0s = y0[idx]
        trend = float(trend_accuracy(yt, yp, y0s))
        corr = float(mean_horizon_correlation(yt, yp))
        comp = float(composite_score(yt, yp, y0s))
        dtrue = yt - y0s.reshape(-1, 1)
        dpred = yp - y0s.reshape(-1, 1)
        th = np.array(_trend_by_horizon(dtrue, dpred), float)
        k15 = float(th[: min(5, len(th))].mean()) if th.size else 0.0
        k1115 = float(th[10:15].mean()) if th.size >= 15 else (float(th[10:].mean()) if th.size > 10 else k15)
        rows.append(
            {
                "symbol": str(sym),
                "n": int(idx.size),
                "trend": trend,
                "corr": corr,
                "composite": comp,
                "trend_k1_5": k15,
                "trend_k11_15": k1115,
                "trend_long_minus_short": float(k1115 - k15),
            }
        )
    if not rows:
        return {"available": False, "reason": "insufficient rows per symbol in preview subset", "note": note}
    sdf = pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)
    delta_dist = sdf["trend_long_minus_short"].to_numpy(dtype=float)
    return {
        "available": True,
        "note": note,
        "n_symbols": int(len(sdf)),
        "top10_by_composite": sdf.head(10).to_dict(orient="records"),
        "bottom10_by_composite": sdf.tail(10).sort_values("composite").to_dict(orient="records"),
        "trend_long_minus_short_distribution": {
            "mean": float(np.mean(delta_dist)) if len(delta_dist) else 0.0,
            "std": float(np.std(delta_dist)) if len(delta_dist) else 0.0,
            "min": float(np.min(delta_dist)) if len(delta_dist) else 0.0,
            "max": float(np.max(delta_dist)) if len(delta_dist) else 0.0,
            "positive_rate": float(np.mean(delta_dist > 0)) if len(delta_dist) else 0.0,
        },
    }


def _try_plots(run_dir: Path, trend_k: list[float], delta_mag_k: list[float], pos_rate_k: list[float]) -> list[str]:
    saved: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return saved
    x = np.arange(1, len(trend_k) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, trend_k, marker="o")
    ax.set_title("Trend accuracy par horizon (diagnostic)")
    ax.set_xlabel("Horizon k")
    ax.set_ylabel("Trend accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = run_dir / "horizon_trend_diagnostic.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    saved.append(p.name)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, delta_mag_k, marker="o")
    ax.set_title("Magnitude moyenne |delta_true| par horizon")
    ax.set_xlabel("Horizon k")
    ax.set_ylabel("mean(|delta_true|)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = run_dir / "horizon_delta_mag.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    saved.append(p.name)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, pos_rate_k, marker="o")
    ax.set_title("Proportion delta_true > 0 par horizon")
    ax.set_xlabel("Horizon k")
    ax.set_ylabel("positive rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = run_dir / "horizon_pos_rate.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    saved.append(p.name)
    return saved


def diagnose_run(run_dir: Path) -> dict[str, Any]:
    config = _safe_load_json(run_dir / "config.json")
    y_val = np.load(run_dir / "y_val.npy")
    y0_val = np.load(run_dir / "y0_val.npy")
    y_pred, pred_file = _load_pred(run_dir, config)
    preview_path = run_dir / "preview_val.csv"
    preview_df = pd.read_csv(preview_path) if preview_path.exists() else None

    y_val = np.asarray(y_val, dtype=float)
    y0_val = np.asarray(y0_val, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_val.ndim != 2:
        raise SystemExit("y_val.npy must be a 2D array")
    if y_pred.ndim != 2:
        raise SystemExit("prediction array must be a 2D array")

    n, h = y_val.shape
    if y_pred.shape != (n, h):
        raise SystemExit(f"Prediction shape mismatch: expected {(n,h)}, got {tuple(y_pred.shape)}")
    if y0_val.shape != (n,):
        raise SystemExit(f"y0_val shape mismatch: expected {(n,)}, got {tuple(y0_val.shape)}")

    finite_before = {
        "y_val": _finite_stats(y_val),
        "y0_val": _finite_stats(y0_val),
        "y_pred": _finite_stats(y_pred),
    }
    y_pred_clean = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    y_val_clean = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
    y0_clean = np.nan_to_num(y0_val, nan=0.0, posinf=0.0, neginf=0.0)

    delta_true = y_val_clean - y0_clean.reshape(-1, 1)
    delta_pred = y_pred_clean - y0_clean.reshape(-1, 1)

    rng = np.random.default_rng(42)
    sample_n = min(100, n)
    sample_idx = rng.choice(n, size=sample_n, replace=False) if n > 0 else np.array([], dtype=int)
    sample_delta_k1 = delta_true[sample_idx, 0] if sample_n else np.array([], dtype=float)

    # Horizon metrics (corr on deltas if delta_target enabled in config)
    use_delta_corr = bool(config.get("delta_target", False))
    trend_k = _trend_by_horizon(delta_true, delta_pred)
    corr_k = _corr_by_horizon(y_val_clean, y_pred_clean, use_delta=use_delta_corr, y0=y0_clean)
    positive_rate_k: list[float] = []
    majority_baseline_trend_k: list[float] = []
    pct_small_delta_k: list[float] = []
    for k in range(h):
        d = delta_true[:, k]
        p = float(np.mean(d > 0)) if d.size else 0.0
        positive_rate_k.append(p)
        majority_baseline_trend_k.append(float(max(p, 1.0 - p)))
        std_k = float(np.std(d)) if d.size else 0.0
        eps_k = 0.01 * std_k if std_k > 0 else 1e-8
        pct_small_delta_k.append(float(np.mean(np.abs(d) < eps_k)) if d.size else 0.0)
    overall_trend = float(trend_accuracy(y_val_clean, y_pred_clean, y0_clean))
    overall_corr = float(mean_horizon_correlation(y_val_clean, y_pred_clean))
    overall_comp = float(composite_score(y_val_clean, y_pred_clean, y0_clean))

    delta_stats = _delta_stats(delta_true)
    per_symbol = (
        _compute_symbol_breakdown(preview_df, y_val_clean, y_pred_clean, y0_clean)
        if preview_df is not None
        else {"available": False, "reason": "preview_val.csv missing"}
    )

    plots_saved = _try_plots(
        run_dir,
        trend_k,
        [delta_stats["by_horizon"][f"k{k+1}"]["mean_abs_delta"] for k in range(h)],
        [delta_stats["by_horizon"][f"k{k+1}"]["positive_rate"] for k in range(h)],
    )

    out = {
        "run_id": run_dir.name,
        "config_summary": {
            "target": config.get("target"),
            "model": config.get("model"),
            "horizon": int(h),
            "delta_target": bool(config.get("delta_target", False)),
            "limit_files": config.get("limit_files"),
        },
        "alignment_checks": {
            "shapes": {"y_val": [int(n), int(h)], "y0_val": [int(n)], "y_pred": [int(n), int(h)]},
            "prediction_file": pred_file,
            "finite_before": finite_before,
            "pred_non_finite_replaced_with_zero": finite_before["y_pred"]["non_finite_count"] > 0,
            "sample_k1_delta_true_stats": {
                "n_samples": int(sample_n),
                "mean": float(np.mean(sample_delta_k1)) if sample_n else 0.0,
                "std": float(np.std(sample_delta_k1)) if sample_n else 0.0,
                "min": float(np.min(sample_delta_k1)) if sample_n else 0.0,
                "max": float(np.max(sample_delta_k1)) if sample_n else 0.0,
                "positive_rate": float(np.mean(sample_delta_k1 > 0)) if sample_n else 0.0,
            },
        },
        "delta_stats": delta_stats,
        "horizon_metrics": {
            "corr_basis": "delta" if use_delta_corr else "absolute",
            "trend_accuracy_k": [float(v) for v in trend_k],
            "corr_k": [float(v) for v in corr_k],
            "positive_rate": [float(v) for v in positive_rate_k],
            "majority_baseline_trend": [float(v) for v in majority_baseline_trend_k],
            "pct_small_delta": [float(v) for v in pct_small_delta_k],
        },
        "overall_metrics": {
            "trend": overall_trend,
            "mean_correlation": overall_corr,
            "composite": overall_comp,
        },
        "per_symbol": per_symbol,
        "plots_saved": plots_saved,
    }
    return out


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else (run_dir / "diagnostics.json")
    diagnostics = diagnose_run(run_dir)
    out_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
