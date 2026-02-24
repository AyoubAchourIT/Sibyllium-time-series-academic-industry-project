<<<<<<< HEAD
# Sibyllium Forecasting Project

Modular training/evaluation scaffold for the Sibyllium hackathon time-series task (H=15) using XLSX OHLCV inputs in `datas/` and a notebook-driven workflow that imports Python modules.

## Structure

- `src/`: reusable package (`data`, `features`, `metrics`, `models`, `validation`) plus `train.py` and `eval.py` CLIs
- `tests/`: unit and smoke tests
- `configs/`: configuration files (JSON/YAML)
- `notebooks/`: optional analysis notebooks (kept empty here; use `Hackathon_Sibyllium.ipynb`)
- `runs/`: generated run artifacts (manifests, metrics, model outputs)
- `datas/`: raw Excel market files (existing project data)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --limit-files 3 --target macd --model gbdt
python -m src.eval
pytest -q
```

## Training Examples

```bash
# GBDT (tabular lag/rolling features)
python -m src.train --limit-files 20 --target macd --model gbdt

# DLinear (windowed univariate target)
python -m src.train --limit-files 20 --target macd --model dlinear --lookback 256 --epochs 20
```

## Notes

- `python -m src.train` computes indicators, builds datasets, trains baselines plus the selected model, and writes metrics/predictions under `runs/`.
- `python -m src.eval` runs a minimal metrics demo; replace with artifact-based evaluation as models are added.
- No notebook is created in `notebooks/`; continue using `Hackathon_Sibyllium.ipynb` to import and run modules.
=======
# Sibyllium-time-series-academic-industry-project
>>>>>>> 9caae60bc2c25aac76903d0ceeb143a48962aa30
