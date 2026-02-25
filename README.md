# Projet de Prévision Sibyllium

## Contexte
Ce dépôt fournit une base modulaire pour le hackathon Sibyllium de prévision de séries temporelles (horizon `H=15`) à partir de fichiers Excel OHLCV (`.xlsx`) stockés dans `datas/`. Le notebook `Hackathon_Sibyllium.ipynb` sert à l'exploration et à l'orchestration, tandis que la logique de calcul, d'entraînement et d'évaluation est dans `src/`.

## Objectif
Prévoir des indicateurs techniques (ex. `macd`, `stoch_k`) sur plusieurs horizons (`t+1` à `t+15`) et comparer des modèles stables (GBDT, DLinear) avec des baselines (`persistence`, `drift`).

## Métriques
Les métriques principales sont :
- `trend_accuracy` : accord de signe entre variation réelle et variation prédite par rapport à `y0`.
- `mean_correlation` : corrélation moyenne entre trajectoires réelles et prédites sur l'horizon.
- `composite` : score combiné (trend + corrélation).

Définition de `corr_basis` (dans `metrics.json`) :
- `corr_basis=\"level\"` : `corr_by_horizon` et `mean_correlation` utilisent la corrélation sur les niveaux (`y`).
- `corr_basis=\"delta\"` : `corr_by_horizon` et `mean_correlation` utilisent la corrélation sur les deltas (`y - y0`).

Résultats agrégés (100 fichiers) :
- `MACD` : trend=`0.6598`, corr=`0.2499`, composite=`0.4549`
- `Stoch_K` : trend=`0.6778`, corr=`0.2562`, composite=`0.4670`

Les métriques par horizon (`k=1..H`) sont enregistrées dans `runs/<RUN_ID>/metrics.json`.

## Structure du repo
- `src/` : modules Python (`data`, `features`, `metrics`, `models`, `validation`, `report`) + CLI (`train.py`, `eval.py`)
- `tests/` : tests unitaires et smoke tests
- `configs/` : configurations locales
- `notebooks/` : notebooks optionnels (vide ici ; utiliser `Hackathon_Sibyllium.ipynb`)
- `datas/` : données brutes locales (non versionnées)
- `runs/` : artefacts d'entraînement/évaluation (non versionnés)

Important : `datas/` et `runs/` sont des répertoires locaux et ne sont pas versionnés dans Git.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --limit-files 3 --target macd --model gbdt
python -m src.eval
pytest -q
```

## Lancer un benchmark stable
Utiliser ces commandes pour le rapport de benchmark (modèles stables uniquement).

```bash
# MACD benchmark (GBDT + delta target)
python -m src.train --limit-files 100 --target macd --model gbdt --delta-target

# Stoch_K benchmark (GBDT + delta target)
python -m src.train --limit-files 100 --target stoch_k --model gbdt --delta-target

# Stoch_K benchmark (LightGBM + delta target)
python -m src.train --data-dir datas --limit-files 300 --target stoch_k --model lightgbm --delta-target --lookback-lags 128

# Stoch_K benchmark (XGBoost + delta target)
python -m src.train --data-dir datas --limit-files 300 --target stoch_k --model xgboost --delta-target --lookback-lags 128
```

## Visualiser les métriques par horizon
```bash
# Plot per-horizon metrics for a completed run
python -m src.report.plot_horizon_metrics --run-dir runs/<RUN_ID>
```

## Comparer les runs
```bash
# Compare recent runs and export a summary CSV
python -m src.report.summarize_runs --runs-dir runs --last 20

# Diagnostic détaillé d'un run (alignement, métriques par horizon, par symbole)
python -m src.report.diagnose_run --run-dir runs/<RUN_ID> --out runs/<RUN_ID>/diagnostics.json

# Table de benchmark multi-runs (comparaison des modèles/runs)
python -m src.report.benchmark_table --runs-dir runs --out runs/benchmark_table.csv
```

## Modèles expérimentaux
Les modèles NeuralForecast (`nhits`, `nbeats`, `tide`, `ensemble_gbdt_nhits`) sont expérimentaux (WIP) et désactivés par défaut dans `src.train`.

Installer les dépendances optionnelles :

```bash
pip install -r requirements.txt -r requirements-neuralforecast.txt
```

Exemple d'exécution explicite (mode expérimental) :

```bash
python -m src.train --model nhits --experimental-neuralforecast --target macd --input-size 128 --max-steps 50 --val-size 120 --step-size 1
```

## Notes
- `python -m src.train` calcule les indicateurs, construit les jeux de données, entraîne les baselines + le modèle choisi, puis écrit les métriques/prédictions dans `runs/`.
- `python -m src.eval` exécute une démo minimale des métriques.
- Aucun notebook n'est généré automatiquement ; utiliser `Hackathon_Sibyllium.ipynb` pour piloter les modules Python.
