# Sibyllium Forecasting Project

Modular training/evaluation scaffold for the Sibyllium hackathon time-series task (H=15) using XLSX OHLCV inputs in `datas/` and a notebook-driven workflow that imports Python modules.

## Current Best Results (100 files)

| Target | Trend | Corr | Composite |
|---|---:|---:|---:|
| MACD | 0.6598 | 0.2499 | 0.4549 |
| Stoch_K | 0.6778 | 0.2562 | 0.4670 |

These are aggregate validation metrics from the current experiments. Per-horizon metrics are also saved in `runs/<RUN_ID>/metrics.json` for plotting and horizon-decay analysis.

## Structure

- `src/`: reusable package (`data`, `features`, `metrics`, `models`, `validation`) plus `train.py` and `eval.py` CLIs
- `tests/`: unit and smoke tests
- `configs/`: configuration files (JSON/YAML)
- `notebooks/`: optional analysis notebooks (kept empty here; use `Hackathon_Sibyllium.ipynb`)
- `runs/`: generated run artifacts (manifests, metrics, model outputs)
- `datas/`: raw Excel market files (existing project data)
- `runs/` and `datas/` are local artifacts/data directories and are not versioned in Git.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --limit-files 3 --target macd --model gbdt
python -m src.eval
pytest -q
```

## Stable Benchmark

Use these commands for the benchmark report (stable, supported models only).

```bash
# MACD benchmark (GBDT + delta target)
python -m src.train --limit-files 100 --target macd --model gbdt --delta-target

# Stoch_K benchmark (GBDT + delta target)
python -m src.train --limit-files 100 --target stoch_k --model gbdt --delta-target

# Plot per-horizon metrics for a completed run
python -m src.report.plot_horizon_metrics --run-dir runs/<RUN_ID>

# Compare recent runs and export a summary CSV
python -m src.report.summarize_runs --runs-dir runs --last 20
```

## Training Examples

```bash
# GBDT (tabular lag/rolling features)
python -m src.train --limit-files 20 --target macd --model gbdt

# DLinear (windowed univariate target)
python -m src.train --limit-files 20 --target macd --model dlinear --lookback 256 --epochs 20

# Plot per-horizon metrics for a completed run
python -m src.report.plot_horizon_metrics --run-dir runs/<RUN_ID>
```

## Experimental (NeuralForecast)

NeuralForecast models (`nhits`, `nbeats`, `tide`, and `ensemble_gbdt_nhits`) are experimental/WIP and disabled by default in `src.train`.

Install optional dependencies first:

```bash
pip install -r requirements.txt -r requirements-neuralforecast.txt
```

Then enable explicitly:

```bash
python -m src.train --model nhits --experimental-neuralforecast --target macd --input-size 128 --max-steps 50 --val-size 120 --step-size 1
```

## Notes

- `python -m src.train` computes indicators, builds datasets, trains baselines plus the selected model, and writes metrics/predictions under `runs/`.
- `python -m src.eval` runs a minimal metrics demo; replace with artifact-based evaluation as models are added.
- No notebook is created in `notebooks/`; continue using `Hackathon_Sibyllium.ipynb` to import and run modules.
