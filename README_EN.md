# Sibyllium Forecasting Project

## Context
This repository provides a modular training/evaluation scaffold for the Sibyllium time-series hackathon (horizon `H=15`) using OHLCV Excel files (`.xlsx`) stored in `datas/`. The `Hackathon_Sibyllium.ipynb` notebook is used for exploration/orchestration, while the implementation for feature engineering, training, and evaluation lives in `src/`.

## Goal
Forecast technical indicators (e.g., `macd`, `stoch_k`) across multiple horizons (`t+1` to `t+15`) and compare stable models (GBDT, DLinear) against baselines (`persistence`, `drift`).

## Metrics
Main metrics:
- `trend_accuracy`: sign agreement between true and predicted changes relative to `y0`.
- `mean_correlation`: average correlation between true and predicted horizon trajectories.
- `composite`: combined score (trend + correlation).

Aggregate results (100 files):
- `MACD`: trend=`0.6598`, corr=`0.2499`, composite=`0.4549`
- `Stoch_K`: trend=`0.6778`, corr=`0.2562`, composite=`0.4670`

Per-horizon metrics (`k=1..H`) are saved in `runs/<RUN_ID>/metrics.json`.

## Repository Structure
- `src/`: Python modules (`data`, `features`, `metrics`, `models`, `validation`, `report`) + CLIs (`train.py`, `eval.py`)
- `tests/`: unit tests and smoke tests
- `configs/`: local configuration files
- `notebooks/`: optional notebooks (empty here; use `Hackathon_Sibyllium.ipynb`)
- `datas/`: local raw data (not versioned)
- `runs/`: local training/evaluation artifacts (not versioned)

Important: `datas/` and `runs/` are local directories and are not versioned in Git.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --limit-files 3 --target macd --model gbdt
python -m src.eval
pytest -q
```

## Run a Stable Benchmark
Use these commands for the benchmark report (stable models only).

```bash
# MACD benchmark (GBDT + delta target)
python -m src.train --limit-files 100 --target macd --model gbdt --delta-target

# Stoch_K benchmark (GBDT + delta target)
python -m src.train --limit-files 100 --target stoch_k --model gbdt --delta-target
```

## Visualize Horizon Metrics
```bash
# Plot per-horizon metrics for a completed run
python -m src.report.plot_horizon_metrics --run-dir runs/<RUN_ID>
```

## Compare Runs
```bash
# Compare recent runs and export a summary CSV
python -m src.report.summarize_runs --runs-dir runs --last 20
```

## Experimental Models
NeuralForecast models (`nhits`, `nbeats`, `tide`, `ensemble_gbdt_nhits`) are experimental (WIP) and disabled by default in `src.train`.

Install optional dependencies:

```bash
pip install -r requirements.txt -r requirements-neuralforecast.txt
```

Example explicit run (experimental mode):

```bash
python -m src.train --model nhits --experimental-neuralforecast --target macd --input-size 128 --max-steps 50 --val-size 120 --step-size 1
```

## Notes
- `python -m src.train` computes indicators, builds datasets, trains baselines plus the selected model, and writes metrics/predictions under `runs/`.
- `python -m src.eval` runs a minimal metrics demo.
- No notebook is generated automatically; use `Hackathon_Sibyllium.ipynb` to orchestrate the Python modules.
