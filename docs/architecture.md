# Architecture Specification: BikeSense AI

## 1. System Overview
BikeSense AI is a closed-loop MLOps system providing 24-hour hourly station departure forecasts for Citi Bike (NYC). The architecture enforces strict lineage, reproducible data versioning, drift detection, and governed model promotion/rollback.

```
+-----------------------------------------------------------------------------------+
|                              1. Data & Automation                                 |
|  Citi Bike S3 (14 mos) + Open-Meteo Weather + US Holidays                         |
|                           │                                                       |
|                           ▼                                                       |
|  DVC Pipeline: Ingest ──► Weather ──► Holidays ──► Feature Engineering           |
|                           │                                                       |
|                           ▼                                                       |
|  Output: data/processed/features_{v1, v2, v3}.parquet (DVC Content-Hashed)        |
+-----------------------------------------------------------------------------------+
                                    │
                                    ▼
+-----------------------------------------------------------------------------------+
|                              2. Modeling & Registry                               |
|  Time-Ordered Split (No Leakage) ──► Train: Baseline, Ridge, LightGBM             |
|                           │                                                       |
|                           ▼                                                       |
|  MLflow Tracking: SQLite backend (sqlite:///mlflow.db)                            |
|  Parameters, Metrics (MAE, RMSE, Shortfall), Git Commit & DVC Hash Pinned         |
|                           │                                                       |
|                           ▼                                                       |
|  MLflow Model Registry ('bikesense-demand-forecaster')                            |
|       @champion ◄─── Promotion Gate (+3% MAE margin) ◄─── @challenger             |
|           │                                                                       |
|           └─── Fast Rollback (<5s alias revert) ──► @previous_champion            |
+-----------------------------------------------------------------------------------+
                                    │
                                    ▼
+-----------------------------------------------------------------------------------+
|                             3. Serving & Monitoring                               |
|  FastAPI Docker Container loads 'models:/bikesense-demand-forecaster@champion'    |
|                           │                                                       |
|                           ▼                                                       |
|  Prediction Logging ──► 24h Later Ground Truth Join ──► Evidently Drift & Alerts  |
|                           │                                                       |
|                           └──────► Trigger Retrain Workflow on Degradation        |
+-----------------------------------------------------------------------------------+
```

## 2. Model Governance via MLflow Aliases
Instead of updating Docker containers or re-deploying code on new model releases, the serving layer resolves the active model dynamically via MLflow Model Registry aliases:

- `@champion`: Production model queried by the FastAPI `/forecast` service.
- `@challenger`: Candidate model undergoing promotion gate validation on held-out months.
- `@previous_champion`: Retained fallback version for instant rollback (<5 seconds).

## 3. Promotion Gate Margin
- A candidate model registered as `@challenger` is evaluated against the current `@champion` on an unbiased held-out month (excluded from training).
- **Promotion Rule:**
  $$\frac{\text{MAE}_{\text{champ}} - \text{MAE}_{\text{chall}}}{\text{MAE}_{\text{champ}}} \ge 3.0\% \quad \text{and} \quad \frac{\Delta \text{Shortfall}}{\text{Shortfall}_{\text{champ}}} \le 5.0\%$$
- Both quantitative criteria and human approval are required before swapping the `@champion` alias.
