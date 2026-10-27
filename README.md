# BikeSense AI: Drift-Aware Bike-Share Demand Forecasting

A versioned and monitored MLOps pipeline that forecasts hourly station-level demand for Citi Bike (New York City) and keeps the forecasting model trustworthy over time.

**Course:** MLOps (final-year project) · **Infrastructure cost:** USD 0 (open data, open-source tools) · **Duration:** 8 weeks

---

## Table of Contents

1. [What this project is](#1-what-this-project-is)
2. [Business significance and its limits](#2-business-significance-and-its-limits)
3. [Forecasting task](#3-forecasting-task)
4. [Architecture](#4-architecture)
5. [Tech stack](#5-tech-stack)
6. [Data sources](#6-data-sources)
7. [Dataset versions](#7-dataset-versions)
8. [Repository layout](#8-repository-layout)
9. [Getting started](#9-getting-started)
10. [Using the service](#10-using-the-service)
11. [Model versioning and promotion](#11-model-versioning-and-promotion)
12. [Monitoring](#12-monitoring)
13. [Demonstration flow](#13-demonstration-flow)
14. [Testing](#14-testing)
15. [CI/CD](#15-cicd)
16. [Success criteria](#16-success-criteria)
17. [Known limitations](#17-known-limitations)
18. [References](#18-references)
19. [Team](#19-team)

---

## 1. What this project is

Bike-share operators must move bikes between stations before demand arrives. A forecasting model trained once degrades as seasons, weather and rider behaviour change, and the degradation stays invisible until predictions are compared with the ground truth that arrives later.

This project builds a **24-hour, station-level departure forecasting service** and, as its core contribution, the controls around it:

| Pillar | Question it answers | Tool |
|---|---|---|
| Data versioning | Can every dataset be reproduced exactly? | DVC + Git |
| Model versioning | Which data and code produced the model now in service? Can we roll back? | MLflow tracking + registry |
| Monitoring | Is the model still accurate this month? | Evidently AI |
| Automation | Can retraining and checks run without manual steps? | CI checks + retrain workflow |

The forecasting model is deliberately simple. The point is the lifecycle around it.

## 2. Business significance and its limits

Stations that are empty lose rentals; stations that are full block returns. Overnight rebalancing plans depend on target inventory levels derived from demand forecasts, so a more trustworthy forecast feeds directly into service availability and operating effort.

What this project **does not** claim:

- **No monetary savings.** Public trip files record rides that happened, not availability. Where a station was empty, demand went unrecorded (censored demand), so the target understates true demand.
- **Value of forecasting is conditional.** Published research asks whether demand forecasting is worth its cost for rebalancing; the benefit depends on how the operator rebalances.
- **What it does claim:** a reproducible, monitored forecasting pipeline evaluated on forecast error and an under-forecast shortfall proxy.

## 3. Forecasting task

- **Target:** departures per station per hour
- **Horizon:** next 24 hours, issued once per day at 00:00 (ground truth is known 24 hours later)
- **Stations:** the 30 busiest, fixed from dataset v1
- **Features:** hour, weekday, month, holiday flag, temperature, precipitation, wind, demand lagged 24 h and 168 h, 7-day same-hour mean (all computed only from information available at issue time)
- **Models:** seasonal-naive baseline (demand one week earlier), Ridge regression, LightGBM (several hyperparameter sets)
- **Validation:** time-ordered splits only, no shuffling
- **Metrics:** MAE (primary), RMSE, under-forecast shortfall (total demand units under-predicted; a proxy for service risk, not a monetary figure)

## 4. Architecture

```
Public data (Citi Bike CSV + Open-Meteo + holidays)
        │
        ▼
Ingest & validate (pandas + schema tests)
        │
        ▼
Data versioning (DVC: dataset v1, v2, v3)
        │
        ▼
Feature engineering (temporal + weather + lag + rolling)
        │
        ▼
Train & evaluate (baseline, Ridge, LightGBM)
        │
        ▼
Model versioning (MLflow registry, champion / challenger aliases)
        │
        ▼
Serving (FastAPI in Docker, dynamically loads the champion alias)
        │
        ▼
Prediction log ──joined with actual demand 24 h later──► Monitoring (Evidently AI)
                                                              │
                                                              ▼
                                                  Rolling error tracking
                                                              │
                                                              ▼
                                                  Alert engine & retrain workflow
```

## 5. Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Data versioning | DVC (local remote) + Git |
| Tracking and registry | MLflow (local server, SQLite backend) |
| Modeling | scikit-learn (Ridge), LightGBM |
| Serving | FastAPI, Uvicorn, Docker Compose |
| Monitoring | Evidently AI (data drift, model performance) |
| Alerting | Custom multi-signal alert engine |
| Automation | GitHub Actions CI + retrain workflow |
| Testing | pytest, ruff (linting) |

## 6. Data sources

| Source | Content | Terms |
|---|---|---|
| Citi Bike trip histories | Trip start/end time, stations, rider type, bike type | Free; NYCBS Data Use Policy |
| Open-Meteo Historical Weather API | Hourly temperature, precipitation, wind | Free for non-commercial use; CC BY 4.0 attribution |
| Python `holidays` package | US public holiday flags | Open source |

Weather responses are fetched once per month-chunk and cached as DVC-tracked files, so results are reproducible and the API is called only a handful of times.

**Attribution:** weather data by [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0).

## 7. Dataset versions

Fourteen consecutive months (M1 to M14, chosen to span a seasonal transition) are released in three steps to mimic data arriving over time.

| Version | Training data | Replay window (treated as production) |
|---|---|---|
| v1 | M1 to M8 | M9 to M10 |
| v2 | M1 to M10 | M11 to M12 |
| v3 | M1 to M12 | M13 to M14 |

## 8. Repository layout

```
.
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── Makefile                          # Convenience targets
├── dvc.yaml                          # DVC pipeline: ingest → features → train
├── dvc.lock                          # Pinned content hashes per dataset version
├── params.yaml                       # Global parameters (dates, stations, etc.)
├── docker-compose.yml                # Compose orchestration for service + MLflow
│
├── configs/
│   ├── features.yaml                 # Feature engineering configuration
│   ├── model.yaml                    # Model hyperparameters (Ridge, LightGBM)
│   ├── gate.yaml                     # Promotion gate thresholds
│   └── monitoring.yaml               # Drift and alerting thresholds
│
├── data/                             # DVC-tracked data directories
│   ├── raw/                          # Ingested Citi Bike trip data
│   ├── external/                     # Weather data, holiday calendar
│   ├── processed/                    # Feature-engineered datasets
│   └── predictions/                  # Prediction logs from service
│
├── src/
│   ├── data/                         # Ingestion, validation, weather, holidays
│   │   ├── ingest.py                 # Citi Bike data downloader
│   │   ├── validate.py               # Schema & quality checks
│   │   ├── weather.py                # Open-Meteo API client
│   │   └── holidays.py               # Holiday calendar builder
│   ├── features/
│   │   └── build_features.py         # Temporal, weather, lag, rolling features
│   ├── models/
│   │   ├── baseline.py               # Seasonal-naive baseline
│   │   ├── train.py                  # Time-ordered training + MLflow logging
│   │   ├── evaluate.py               # Evaluation report generator
│   │   └── metrics.py                # MAE, RMSE, shortfall computations
│   ├── registry/
│   │   ├── register.py               # Register best run as challenger
│   │   ├── promote.py                # Promotion gate (champion vs challenger)
│   │   └── rollback.py               # Instant champion rollback
│   ├── replay/
│   │   └── replay_driver.py          # Simulated production replay
│   └── monitoring/
│       ├── join_actuals.py            # 24h delayed ground truth join
│       ├── rolling_error.py           # Rolling 7-day MAE tracker
│       ├── drift_report.py            # Evidently AI drift report generator
│       └── alerts.py                  # Multi-signal alert engine
│
├── service/                           # FastAPI microservice
│   ├── Dockerfile                     # Container specification
│   └── app/
│       ├── main.py                    # FastAPI app (/health, /model-info, /forecast)
│       ├── schemas.py                 # Pydantic request/response models
│       ├── model_loader.py            # Dynamic champion loader from MLflow
│       └── prediction_log.py          # Thread-safe prediction logger
│
├── tests/                             # 38 unit and integration tests
│   ├── data/                          # Schema and validation tests
│   ├── features/                      # Feature engineering tests
│   ├── models/                        # Metrics, baseline, training tests
│   ├── registry/                      # Promotion gate and rollback tests
│   ├── service/                       # API endpoint and prediction log tests
│   └── monitoring/                    # Alerts, join, rolling error tests
│
├── scripts/
│   ├── run_demo.sh                    # End-to-end demonstration script
│   ├── registry_audit.py              # Lineage audit for all registered versions
│   └── fresh_clone_test.sh            # Reproducibility verification
│
├── docs/
│   ├── architecture.md                # System architecture documentation
│   ├── model_card.md                  # Model card (inputs, outputs, metrics)
│   ├── data_card.md                   # Data card (sources, schema, versions)
│   ├── lineage_report.md              # End-to-end lineage documentation
│   └── monitoring_guide.md            # Operational monitoring runbook
│
├── reports/                           # Generated evaluation and monitoring outputs
│
└── .github/
    ├── CODEOWNERS                     # Code ownership by team member
    └── workflows/
        ├── ci.yml                     # Lint + test on every push/PR
        └── retrain.yml                # Automated retrain on dataset update
```

## 9. Getting started

**Prerequisites:** Python 3.10+, Docker and Docker Compose, Git, DVC, about 10 GB free disk.

```bash
# 1. Clone and set up environment
git clone https://github.com/Emperor2004/bikesense-ai.git
cd bikesense-ai
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Reproduce data and model pipeline for the current dataset version
dvc repro

# 3. Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

# 4. Start the forecasting service
docker compose up --build
```

**Convenience targets (Makefile):**

```bash
make data         # run ingest + features
make train        # train baseline, Ridge, LightGBM and log to MLflow
make register     # register best run as challenger
make serve        # start the FastAPI service
make replay       # replay the next window as simulated production
make monitor      # generate Evidently drift reports
make test         # run ruff linting + pytest
make demo         # run the end-to-end demonstration
make clean        # remove generated data files
```

## 10. Using the service

```bash
# Health check — returns service status, loaded champion version, and uptime
curl http://localhost:8000/health

# 24-hour forecast for a specific station and date
curl "http://localhost:8000/forecast?station_id=<STATION_ID>&date=<YYYY-MM-DD>"

# Model lineage — which model is serving, and where did it come from?
curl http://localhost:8000/model-info
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health, champion version, uptime |
| `/model-info` | GET | Champion version, Git commit, DVC dataset hash, run ID |
| `/forecast` | GET | 24-hour station-level departure forecast |

Every prediction is appended to a thread-safe prediction log (`data/predictions/prediction_log.csv`) and later joined with actual demand for monitoring.

## 11. Model versioning and promotion

- Every training run logs parameters, metrics, feature list, Git commit, dataset version (DVC hash) and the model artifact to MLflow.
- The best run is registered as **challenger**. The service dynamically loads the **champion** alias, so promotion and rollback are alias changes — not code redeployments.
- **Promotion gate:** a challenger replaces the champion only if its MAE on the latest held-out month is lower by a configured margin (default 3%, set in `configs/gate.yaml`). A shortfall guardrail prevents promotion if under-forecast risk increases.

```bash
# Promote a challenger to champion (runs gate checks automatically)
python -m src.registry.promote --challenger <VERSION>

# Instant rollback — repoints the champion alias, no redeploy needed
python -m src.registry.rollback --to <VERSION>

# Audit lineage for all registered model versions
python scripts/registry_audit.py
```

## 12. Monitoring

During replay the service issues a forecast each day. Actual demand is joined to the prediction log 24 hours later. The reference dataset is the training data of the current champion.

| Layer | Signal | Method | Action |
|---|---|---|---|
| Input drift | Weather, calendar, lag features vs reference | Evidently AI data drift report (KS/Wasserstein) | `ALERT-101`: Flag when share of drifted features > 30% |
| Model performance | Rolling 7-day MAE vs reference MAE | Custom rolling error tracker | `ALERT-102`: Alert when MAE > 1.25× reference |
| Operational risk | Under-forecast shortfall surge | Actuals join pipeline | `ALERT-103`: Alert on shortfall spike |

**Monitoring outputs:**
- Evidently HTML reports → `reports/monitoring/`
- Drift summary JSON → `reports/monitoring/drift_summary.json`
- Active alerts → `reports/monitoring/active_alerts.json`
- Rolling error metrics → `reports/monitoring/rolling_performance.csv`

All monitoring thresholds are configurable in `configs/monitoring.yaml`.

For detailed operational procedures, see the [Monitoring Guide](docs/monitoring_guide.md).

## 13. Demonstration flow

1. **Champion serves forecasts** — trained on v1; show its lineage (Git commit and DVC data hash) via `/model-info`.
2. **Replay production window** — monitoring detects input drift and rising rolling MAE.
3. **Release dataset v2 and retrain** — compare challenger and champion in MLflow UI.
4. **Apply the promotion gate** — switch the champion alias; the service hot-loads the new model without restart.
5. **Roll back** — repoint the alias and verify recovery without a code change or redeployment.

```bash
# Run the full end-to-end demo
bash scripts/run_demo.sh
```

## 14. Testing

The project includes **38 tests** covering all three stages:

| Test Module | Tests | Scope |
|---|---|---|
| `tests/data/` | 7 | Schema validation, date checks, null/negative guards, station coverage |
| `tests/features/` | 5 | Temporal features, lag leakage prevention, rolling means, holiday flags |
| `tests/models/` | 9 | MAE/RMSE/shortfall metrics, baseline predictions, time-ordered splits, lineage helpers |
| `tests/registry/` | 6 | Promotion gate (approval, rejection, shortfall guardrail), rollback scenarios |
| `tests/service/` | 5 | Health, model-info, forecast endpoints, invalid input handling, prediction logging |
| `tests/monitoring/` | 6 | Alert triggers, actuals join, rolling error computation |

```bash
# Run the full test suite with linting
make test

# Or individually
ruff check src/ tests/       # Lint
pytest tests/ -v --tb=short  # Tests
```

## 15. CI/CD

Two GitHub Actions workflows automate quality and retraining:

| Workflow | Trigger | Steps |
|---|---|---|
| `ci.yml` | Every push and pull request | Install dependencies → ruff lint → pytest |
| `retrain.yml` | Dataset version update (manual dispatch) | `dvc repro` → train → evaluate → register challenger |

Code ownership is enforced via `.github/CODEOWNERS`, mapping each stage to its responsible team member.

## 16. Success criteria

| Criterion | Target |
|---|---|
| Forecast quality | Champion beats the seasonal-naive baseline on MAE on each replay window |
| Business proxy | Under-forecast shortfall reported next to MAE for every window and model version |
| Lineage | 100% of registered model versions carry Git commit and dataset version |
| Reproducibility | A clean copy running `dvc repro` reproduces registered metrics within a small tolerance |
| Detection | An alert fires within one monitoring window (7 days) of a visible degradation |
| Rollback | Reverting the champion alias restores service in under 5 minutes, no redeploy |

## 17. Known limitations

- **Censored demand:** trip data understates demand where stations were empty.
- **Weather optimism:** historical weather is reanalysis (ERA5-based), which a live system would replace with forecasts, so accuracy is likely optimistic.
- **Drift depends on the chosen months:** observed drift is reported honestly; if real drift is weak, a clearly labelled injected shift is used as a secondary test only.
- **Local deployment only:** no Kubernetes, cloud, feature store, streaming feed, rebalancing optimisation or deep learning.

## 18. References

- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS 28.
- Citi Bike System Data and NYCBS Data Use Policy: https://citibikenyc.com/system-data
- Open-Meteo Historical Weather API (CC BY 4.0): https://open-meteo.com
- Ataç, S., Obrenović, N., Bierlaire, M. (2020). *Vehicle sharing systems: Does demand forecasting yield a better service?* STRC.
- *Predictive and Prescriptive Performance of Bike-Sharing Demand Forecasts for Inventory Management.* arXiv:2108.00858.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *QJRMS*, 146(730), 1999-2049.
- Official documentation: [DVC](https://dvc.org/doc), [MLflow](https://mlflow.org/docs/latest/index.html), [Evidently AI](https://docs.evidentlyai.com/).

## 19. Team

| Member | Role | Focus |
|---|---|---|
| Om Narayan Pandit | Data and Automation | Ingestion, features, DVC pipeline, dataset versions, replay, CI |
| Pranjali Vishwakarma | Modeling and Registry | Models, evaluation, MLflow registry, promotion gate, rollback |
| Rajat Dwivedi | Serving and Monitoring | FastAPI service, Docker, prediction log, Evidently, alerts |
