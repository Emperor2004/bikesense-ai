# BikeSense AI — Comprehensive Project Description & Technical Specification

## 1. Executive Summary

**BikeSense AI** is an enterprise-grade, end-to-end Machine Learning Operations (MLOps) system designed to solve the station-level bicycle rebalancing challenge in the Citi Bike network (Jersey City / New York City). 

Bike-share systems experience severe operational inefficiencies when stations either run completely empty (preventing departures and resulting in lost ridership) or become completely full (preventing returns and frustrating commuters). Effective overnight and intra-day rebalancing relies heavily on accurate, 24-hour lookahead departure forecasts. However, static models deployed into production experience rapid performance degradation due to seasonality, sudden weather shifts, holiday schedules, and evolving rider behavior.

BikeSense AI addresses this challenge not just with predictive modeling, but by providing **a closed-loop, drift-aware, reproducible MLOps lifecycle**:
- **Dataset Versioning:** Multi-year data pipeline versioned with DVC and backed by Amazon S3 (`s3://om-dvc-bucket`).
- **Feature Engineering:** Strict time-ordered feature store capturing diurnal patterns, weather conditions, holidays, and autoregressive lag indicators.
- **Experiment Tracking:** DVC experiment sweeps (`dvc exp run`) coupled with MLflow tracking.
- **Model Registry & Zero-Downtime Serving:** Dynamic `@champion` resolution enabling seamless hot-reloads in production without service interruption.
- **Production Microservice:** A modern FastAPI application coupled with an interactive Chart.js frontend dashboard, packaged into an Amazon EC2-compatible Docker container.
- **Continuous Delivery:** Robust CI/CD workflows executing automated linting, tests, DVC reproduction, container publishing to Amazon ECR, EC2 automated deployment, and weekly scheduled retraining.

---

## 2. Problem Formulation & Mathematical Objectives

### 2.1 The Forecasting Task
* **Target Variable ($y_{s,t}$):** Number of bike departures from station $s \in \{1, \dots, 30\}$ during hourly interval $t$.
* **Forecast Horizon:** 24 consecutive hours ($t+1, \dots, t+24$) issued daily at 00:00 UTC.
* **Information Boundary:** For any forecast generated at timestamp $T$, features may only utilize information available strictly prior to $T$.

### 2.2 Evaluation Metrics
The pipeline computes three complementary metrics on all evaluation sets:

1. **Mean Absolute Error (Primary Metric):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   Measures expected absolute deviation in hourly bike departures per station.

2. **Root Mean Squared Error (Outlier Sensitivity):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
   Penalizes large forecasting errors during peak commute spikes.

3. **Under-Forecast Shortfall (Operational Risk Proxy):**
   $$\text{Shortfall} = \sum_{i=1}^N \max(0, y_i - \hat{y}_i)$$
   Under-predicting demand causes station starvation (empty stations). This asymmetric metric measures the cumulative unpredicted demand units, directly proxying lost rental opportunities.

---

## 3. High-Level Architecture & End-to-End Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION & VERSIONING                     │
│  Citi Bike Trip Data (S3) + Open-Meteo Weather API + US Federal Holidays│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         DATA VALIDATION & QC                           │
│  Schema assertion, non-negative counts, date boundaries, zero-fill     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      FEATURE STORE (DVC VERSIONED)                     │
│  - Temporal: hour, day_of_week, month, is_holiday                      │
│  - Meteorological: temperature_2m, precipitation, wind_speed_10m       │
│  - Autoregressive: lag_24h, lag_168h (same hour last week)             │
│  - Rolling Statistics: rolling_7d_same_hour_mean                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        MODEL TRAINING & TUNING                         │
│  - Seasonal-Naive Baseline (168-hour seasonal persistence)             │
│  - Regularized Ridge Regression                                        │
│  - LightGBM Gradient Boosting (Shallow, Deep, Regularized, Tuned)      │
│  - Hyperparameter Search via sklearn ParameterSampler & DVC Exp        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    MLFLOW TRACKING & MODEL REGISTRY                    │
│  - Artifact, metric, and tag logging (Git commit, DVC data hash)       │
│  - Aliases: @challenger, @champion, @previous_champion                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       AUTOMATED PROMOTION GATE                         │
│  Requires MAE improvement >= 3.0% and Shortfall increase <= 5.0%       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION SERVING MICROSERVICE                    │
│  - FastAPI REST API (/health, /model-info, /forecast, /predict)        │
│  - Interactive Chart.js Web Dashboard (served from /)                  │
│  - DynamicModelLoader: Zero-downtime hot-reload on alias update        │
│  - Self-contained Docker Container deployed to Amazon EC2              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   MONITORING & CONTINUOUS RETRAINING                   │
│  - 24-hour delayed actuals join for ground-truth performance tracking  │
│  - Evidently AI covariate and prediction drift detection               │
│  - GitHub Actions CI/CD (lint, test, build, deploy, scheduled retrain) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Engineering & Feature Pipeline

### 4.1 Data Sources
1. **Citi Bike Trip Histories:** Public monthly CSV archives from NYCBS/Lyft S3 buckets containing millions of ride records with start/end timestamps, station IDs, and coordinates.
2. **Open-Meteo Historical Weather API:** High-resolution ERA5-based reanalysis fetching hourly temperature at 2m ($^\circ\text{C}$), precipitation ($\text{mm}$), and wind speed at 10m ($\text{km/h}$) for coordinates $(40.7128^\circ\text{N}, -74.0060^\circ\text{W})$. Cached locally as DVC-tracked parquet partitions.
3. **Holiday Calendars:** Python `holidays` library dynamically mapping federal and state public holidays across the multi-year timeline.

### 4.2 Top Station Selection
To ensure stationary spatial coverage, the system filters all operations to the top 30 stations ranked by cumulative departure volume during the initial training window (`v1`). This list is stored in `data/processed/station_list.json` and kept invariant across progressive dataset versions.

### 4.3 Hourly Grid Construction & Zero-Filling
A common trap in mobility data is neglecting hours with zero departures. BikeSense AI builds a full Cartesian product of all 30 stations $\times$ all 8,760 hours per year, performing a left-join against aggregated ride departures and filling unobserved intervals with 0.

### 4.4 Feature Specifications

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `hour` | Integer [0–23] | Diurnal cycle indicator |
| `day_of_week` | Integer [0–6] | Weekly periodicity (Monday=0) |
| `month` | Integer [1–12] | Macro seasonal index |
| `is_holiday` | Binary [0, 1] | US Federal holiday indicator |
| `temperature_2m` | Float | Ambient temperature ($^\circ\text{C}$) |
| `precipitation` | Float | Rain / snow volume ($\text{mm}$) |
| `wind_speed_10m` | Float | Wind velocity ($\text{km/h}$) |
| `lag_24h` | Float | Departure count at the exact same hour yesterday |
| `lag_168h` | Float | Departure count at the exact same hour last week |
| `rolling_7d_same_hour_mean`| Float | 7-day average of demand at that specific hour |

---

## 5. Dataset Versioning & Rolling Window Strategy

The pipeline supports multi-year progressive validation over 38 months (July 2023 through August 2026), structured as expanding historical training blocks evaluated on subsequent 2-month holdout replay windows:

| Version | Training Span | Training Months | Replay Window | Purpose |
| :---: | :---: | :---: | :---: | :--- |
| **`v1`** | 8 months | Jul 2023 – Feb 2024 | Mar 2024 – Apr 2024 | Initial baseline model development |
| **`v2`** | 10 months | Jul 2023 – Apr 2024 | May 2024 – Jun 2024 | First seasonal expansion |
| **`v3`** | 12 months | Jul 2023 – Jun 2024 | Jul 2024 – Aug 2024 | Full annual cycle capture |
| **`v4`** | 24 months | Jul 2023 – Jun 2025 | Jul 2025 – Aug 2025 | Multi-year summer comparison |
| **`v5`** | 30 months | Jul 2023 – Dec 2025 | Jan 2026 – Feb 2026 | Deep winter seasonality adaptation |
| **`v6`** | 34 months | Jul 2023 – Apr 2026 | May 2026 – Jun 2026 | Spring acceleration window |
| **`v7`** | **36 months** | **Jul 2023 – Jun 2026** | **Jul 2026 – Aug 2026** | **Latest active production dataset** |

---

## 6. Model Suite & Hyperparameter Search

### 6.1 Candidate Algorithms
1. **Seasonal-Naive Baseline:** Predicts departures exactly matching the observed departures from 168 hours prior (same station, same hour, one week ago).
2. **Ridge Regression:** L2-regularized linear model trained on standardized continuous and one-hot categorical features.
3. **LightGBM Regressors:** Highly optimized gradient boosted decision trees supporting fast histogram-based splits, non-linear weather interactions, and complex calendar dependencies.

### 6.2 Hyperparameter Tuning Space (`configs/model.yaml`)
Automated exploration is driven by `sklearn.model_selection.ParameterSampler` across the following distribution space:
* `n_estimators`: `[100, 150, 200, 250]`
* `learning_rate`: `[0.02, 0.04, 0.06, 0.08]`
* `num_leaves`: `[31, 45, 63, 80]`
* `max_depth`: `[5, 6, 7, 8]`
* `reg_alpha` (L1): `[0.0, 0.1, 0.5]`
* `reg_lambda` (L2): `[0.0, 0.1, 0.5]`
* `subsample`: `[0.8, 1.0]`
* `colsample_bytree`: `[0.8, 1.0]`

### 6.3 DVC Experiment Automation (`scripts/run_experiments_and_apply.py`)
Multiple experiment runs are coordinated via `dvc exp run -S <params>`. The runner automatically evaluates all candidate configurations, displays a comparative validation table, applies the winning parameter set to the workspace via `dvc exp apply`, and triggers registration into MLflow.

---

## 7. Model Governance & Production Promotion Gate

### 7.1 Lineage Tracking
Every registered model in MLflow stores strict immutable metadata tags:
* `git_commit`: SHA-1 hash of the exact source code commit.
* `dvc_dataset_hash`: Content hash of the feature parquet file from `dvc.lock`.
* `dataset_version`: Active version tag (e.g. `v7`).
* `run_id`: MLflow run identifier containing serialized model artifacts and training metrics.

### 7.2 Quantitative Promotion Criteria
New model versions are initially assigned the `@challenger` alias. To be promoted to `@champion`, the challenger must pass automated gate checks on the holdout evaluation partition (`configs/gate.yaml`):
1. **MAE Improvement:** Challenger must achieve at least a **3.0% reduction** in MAE over the incumbent champion.
2. **Shortfall Guardrail:** Under-forecast shortfall must not increase by more than **5.0%** to avoid operational service degradation.

When the criteria are met, the challenger is dynamically elevated to `@champion`, while the displaced model is retained as `@previous_champion` for instant zero-code rollback capabilities.

---

## 8. Serving Infrastructure & Frontend Dashboard

### 8.1 REST API Specification (`app/main.py`)
* `GET /`: Serves the interactive HTML5/JavaScript dashboard.
* `GET /health`: Returns service health, uptime, and active `@champion` version.
* `GET /model-info`: Returns end-to-end lineage tags for the currently loaded model.
* `GET /forecast?station_id={id}&date={YYYY-MM-DD}`: Returns a 24-hour departure forecast array with hourly breakdown and daily aggregate.
* `POST /predict`: Direct low-level scoring endpoint accepting custom feature payloads.

### 8.2 Interactive Web Dashboard (`app/static/index.html`)
The frontend dashboard provides operators with real-time visual insights:
* **Real-Time Lineage Badge:** Displays the active champion version, dataset hash, and commit hash.
* **Interactive Station & Date Selectors:** Instant dynamic querying across all 30 top stations.
* **24-Hour Departure Forecast Chart:** Interactive Chart.js time-series visualization showing predicted demand trajectories.
* **Summary KPI Cards:** Total daily predicted departures, peak morning rush hour, and peak evening rush hour metrics.

### 8.3 Zero-Downtime Hot-Reloading (`service/app/model_loader.py`)
The `DynamicModelLoader` polls or queries the MLflow Model Registry. When the `@champion` alias is updated in the registry, the microservice automatically reloads the new artifact in memory without dropping in-flight HTTP connections or requiring a container restart.

---

## 9. DevOps, Containerization & CI/CD Workflows

### 9.1 Containerization (`Dockerfile`)
Built on `python:3.11-slim`, incorporating OpenBLAS concurrency safeguards (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`), pre-cached dependencies, static frontend assets, and production Uvicorn ASGI configuration.

### 9.2 GitHub Actions Pipelines
1. **Continuous Integration (`.github/workflows/ci.yml`):**
   * Executes on every push and pull request to `main`.
   * Enforces Ruff code formatting, flake8 linting, and mypy type checks.
   * Runs the 38-case test suite (`pytest tests/`).
   * Verifies DVC pipeline integrity via dry-run reproduction checks.
2. **Continuous Deployment (`.github/workflows/cd.yml`):**
   * Triggers automatically upon merges to `main`.
   * Authenticates with AWS, builds the multi-stage Docker image, and pushes to Amazon ECR.
   * Connects via SSH to target Amazon EC2 instances, pulls the latest image, and restarts the container using Docker Compose with zero manual intervention.
3. **Automated Retraining (`.github/workflows/retrain.yml`):**
   * Configured with a scheduled weekly cron trigger (`0 2 * * 0`) and manual workflow dispatch.
   * Pulls latest dataset artifacts from Amazon S3 DVC remote.
   * Executes `dvc repro`, registers candidate models, runs promotion gate checks, and pushes updated weights.

---

## 10. Operational Monitoring & Drift Detection

1. **Delayed Ground Truth Join (`src/monitoring/join_actuals.py`):**
   Historical predictions logged to `data/predictions/prediction_log.csv` are joined with true Citi Bike ride departures after a 24-hour lag.
2. **Evidently AI Drift Reports (`src/monitoring/drift_report.py`):**
   Computes Kolmogorov-Smirnov and Wasserstein distance tests across weather and lag feature distributions comparing production inference windows against champion training data.
3. **Multi-Signal Alert Engine (`src/monitoring/alerts.py`):**
   * `ALERT-101`: Triggered when > 30% of input features exhibit statistical distribution drift.
   * `ALERT-102`: Triggered when rolling 7-day MAE exceeds 1.25× baseline reference MAE.
   * `ALERT-103`: Triggered when under-forecast shortfall surges, indicating severe stockout risk.

---

## 11. Current Performance Benchmark

Evaluated on the 36-month dataset (`v7`) holdout replay window:

| Model Version | Architecture | Holdout MAE | RMSE | Shortfall (Units) | Promotion Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | Seasonal-Naive (168h) | 1.8410 | 2.6512 | 19,840.00 | Reference |
| **Version 1** | LightGBM (`lgbm_shallow`) | 1.3864 | 2.0411 | 13,934.50 | Displaced Champion |
| **Version 2** | LightGBM (`lgbm_tune_trial_3`) | 1.3992 | 2.0513 | 13,963.28 | Rejected (Insufficient gain) |
| **Version 3** | **LightGBM (`lgbm_tune_trial_2`)** | **1.3153** | **2.0037** | **14,179.34** | 🏆 **ACTIVE @champion (+5.13% gain)** |
