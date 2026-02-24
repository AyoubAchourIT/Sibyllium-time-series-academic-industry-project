# Sibyllium Hackathon — Time Series Forecasting (H=15)

## Goal
Forecast financial indicators (MACD(12,26,9) and Stochastic(14,3,5)) at horizon H=15.
Evaluation uses:
1) Trend accuracy (direction correctness)
2) Correlation (shape similarity)

## Non-negotiables
- No data leakage: all features must use only past information (shift before rolling).
- Time-based splits only (no shuffle).
- Reproducibility: fixed seeds, configs, deterministic settings when possible.
- Output is multi-horizon: predict a 15-step vector for each target series.

## Data reality (important)
- Raw inputs are many XLSX files with intraday OHLCV columns: date, open, high, low, close, volume.
- Compute MACD(12,26,9) and Stochastic(14,3,5) from OHLCV (do not assume indicators are provided).
- Horizon H=15 is in “bars” (15 steps ahead at the native sampling frequency).

## Repository structure
- src/
  - data/ (load, clean, split)
  - features/ (lag/rolling features with safe shifting)
  - metrics/ (trend + correlation)
  - models/ (baselines + main ML + optional deep)
  - validation/ (walk-forward / rolling CV)
  - train.py / predict.py
- notebooks/ (exploration + final report notebook)
- configs/ (yaml or json)
- tests/ (small unit tests for leakage + metrics)
- README.md (setup + how to run)

## Must-have baselines
- Persistence: y_hat[t+k] = y[t]
- Drift/EMA baseline: extrapolate using recent slope or EMA
Optional: ARIMA if easy

## Main model (recommended)
- Multi-output gradient boosting (LightGBM/XGBoost/CatBoost)
- Predict deltas relative to current value: d[t+k] = y[t+k] - y[t]
- Select model by composite score = 0.5*trend + 0.5*corr

## Metrics definitions
- Trend: compare sign(y[t+k]-y[t]) vs sign(yhat[t+k]-y[t]) for k=1..15; average over k and over samples.
- Corr: Pearson correlation between vectors y[t+1..t+15] and yhat[t+1..t+15], average over samples.

## Coding style
- Python 3.11, type hints, docstrings
- Use pandas/numpy/sklearn; prefer lightgbm if available
- Keep functions pure and testable
- Log key outputs and save artifacts under runs/

## Deliverables
- Final notebook reproducing results end-to-end
- CLI training + evaluation script
- Clear result table baselines vs model