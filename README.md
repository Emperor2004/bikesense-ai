# 🚲 BikeSense AI: Drift-Aware Bike-Share Demand Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![DVC](https://img.shields.io/badge/DVC-3.48+-9cf.svg?logo=dvc&logoColor=white)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.12+-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECR%20%26%20EC2-FF9900.svg?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF.svg?logo=github-actions&logoColor=white)](https://github.com/features/actions)

A complete, production-grade MLOps system that forecasts hourly station-level bicycle departures for Citi Bike (New York / Jersey City) across **38 months (July 2023 – August 2026)**, featuring automated DVC experiment sweeps, MLflow Model Registry governance with zero-downtime dynamic serving, an interactive web dashboard, and automated AWS EC2 CI/CD deployment.

> 📖 **Looking for full mathematical formulas, feature engineering details, and technical specifications?**  
> See the [Comprehensive Project Description](Project_Description.md).

---

## 📌 Table of Contents

1. [System Overview & Business Value](#1-system-overview--business-value)
2. [Interactive Web Dashboard & Endpoints](#2-interactive-web-dashboard--endpoints)
3. [Architecture & Pipeline Design](#3-architecture--pipeline-design)
4. [Dataset Versions & Rolling Windows](#4-dataset-versions--rolling-windows)
5. [DVC Hyperparameter Experimentation](#5-dvc-hyperparameter-experimentation)
6. [Model Registry & Dynamic Promotion Gate](#6-model-registry--dynamic-promotion-gate)
7. [Repository Layout](#7-repository-layout)
8. [Quickstart: Running Locally](#8-quickstart-running-locally)
9. [Running with Docker & Cloud Deployment](#9-running-with-docker--cloud-deployment)
10. [Continuous Integration & Continuous Deployment (CI/CD)](#10-continuous-integration--continuous-deployment-cicd)
11. [Monitoring & Drift Detection](#11-monitoring--drift-detection)
12. [Benchmark Results](#12-benchmark-results)
13. [Testing & Quality Assurance](#13-testing--quality-assurance)
14. [Team & License](#14-team--license)

---

## 1. System Overview & Business Value

In urban bike-share networks, demand imbalances create severe service disruptions: **empty stations** leave commuters stranded with lost revenue, while **full stations** block riders from returning bicycles. Accurate 24-hour departure forecasts allow dispatchers to proactively schedule truck-based rebalancing during off-peak hours.

However, static machine learning models degrade quickly under changing weather, holiday schedules, transit disruptions, and long-term ridership shifts. **BikeSense AI** delivers a self-healing, fully governed production pipeline:

| Capability | Problem It Solves | Implementation |
| :--- | :--- | :--- |
| **Data Versioning** | Dataset drift and non-reproducible feature sets | DVC with Amazon S3 Remote (`s3://om-dvc-bucket`) |
| **Hyperparameter Sweeps** | Finding optimal tree depth, leaf count, and learning rates | `dvc exp run` + scikit-learn `ParameterSampler` |
| **Model Registry** | Model lineage, artifact provenance, and rollback safety | MLflow Registry with `@champion` and `@challenger` aliases |
| **Zero-Downtime Serving** | Container restarts during model promotion cause dropped requests | `DynamicModelLoader` in FastAPI dynamically reloads models in memory |
| **Real-Time Observability** | Operators need visual access to predictions and lineage | Interactive Chart.js Web Dashboard served at `/` |
| **Automated Retraining & CD**| Manual deploys and stale production models | GitHub Actions: Weekly retrain cron + Automated AWS ECR/EC2 CD |

---

## 2. Interactive Web Dashboard & Endpoints

The FastAPI microservice in [app/main.py](app/main.py) serves an interactive operational dashboard directly from the root `/` endpoint, alongside documented RESTful API endpoints.

```
┌────────────────────────────────────────────────────────────────────────┐
│  🚲 BikeSense AI — Citi Bike Demand Forecasting Dashboard             │
│  Serving Champion: Version 3  |  Dataset: v7  |  Lineage: [f2f6d16...] │
├────────────────────────────────────────────────────────────────────────┤
│  Station: [ Grove St PATH (JC013) ▼ ]    Date: [ 2026-06-15 ] [Forecast]│
├────────────────────────────────────────────────────────────────────────┤
│  [ Total Departures: 412 ]   [ Morning Peak: 08:00 ]   [ Evening: 18:00 ]│
│                                                                        │
│  Hourly Departure Trajectory:                                          │
│   40 ┤                     ╭──╮                                         │
│   30 ┤       ╭─╮           │  │                                         │
│   20 ┤       │ │   ╭───╮   │  │                                         │
│   10 ┤   ╭───╯ ╰───╯   ╰───╯  ╰───╮                                     │
│    0 ┴───┴────────────────────────┴──► Hour (00:00 - 23:00)            │
└────────────────────────────────────────────────────────────────────────┘
```

### API Endpoints Overview

| Endpoint | Method | Response Model | Description |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | HTML5/JS | Interactive forecasting dashboard with live Chart.js graphs |
| `/health` | `GET` | `HealthResponse` | Service health status, uptime, and currently active champion version |
| `/model-info`| `GET` | `ModelInfoResponse` | Exact lineage tags (Git commit SHA, DVC data hash, run ID, loaded timestamp) |
| `/forecast` | `GET` | `ForecastResponse` | 24-hour departure forecast for specified `station_id` and `date` |
| `/predict` | `POST`| `PredictionResponse` | Real-time scoring endpoint accepting custom feature payloads |
| `/docs` | `GET` | OpenAPI/Swagger | Interactive API specification and testing console |

---

## 3. Architecture & Pipeline Design

The system coordinates data extraction, feature generation, training, evaluation, serving, and monitoring through an automated DVC pipeline:

```
[ Citi Bike Trip CSVs ] ──┐
[ Open-Meteo Weather  ] ──┼─► Ingestion & QC (src/data/ingest.py) ──► data/raw/departures_hourly.parquet
[ US Holiday Calendar ] ──┘                 │
                                            ▼
                              Feature Store (src/features/build_features.py)
                              (24h/168h Lags, 7d Rolling Means, Weather, Holidays)
                                            │
                                            ▼
                              Multi-Model Training (src/models/train.py)
                              ├── Baseline (168-hour seasonal persistence)
                              ├── Ridge Regression (L2 regularization)
                              └── LightGBM Gradient Boosters (Tuned configurations)
                                            │
                                            ▼
                              MLflow Experiment Tracking & Registry
                              (Lineage tags: Git Commit + DVC Hash + Artifacts)
                                            │
                                            ▼
                              Automated Promotion Gate (src/registry/promote.py)
                              (MAE Gain >= 3.0%, Shortfall Delta <= 5.0%)
                                            │
                                            ▼
                              FastAPI Serving & Web Dashboard (app/main.py)
                              (Zero-Downtime Dynamic Model Hot-Reloading)
                                            │
                                            ▼
                              Monitoring & Continuous Delivery (GitHub Actions)
                              (Evidently AI Drift Reports + AWS ECR/EC2 Deployment)
```

---

## 4. Dataset Versions & Rolling Windows

The pipeline versions 38 continuous months of data (July 2023 – August 2026), capturing seasonal transitions, severe weather events, and network growth:

| Version | Training Range | Training Span | Holdout Replay Window | Purpose |
| :---: | :---: | :---: | :---: | :--- |
| `v1` | Jul 2023 – Feb 2024 | 8 months | Mar 2024 – Apr 2024 | Initial baseline model development |
| `v2` | Jul 2023 – Apr 2024 | 10 months | May 2024 – Jun 2024 | First seasonal expansion |
| `v3` | Jul 2023 – Jun 2024 | 12 months | Jul 2024 – Aug 2024 | Complete annual cycle |
| `v4` | Jul 2023 – Jun 2025 | 24 months | Jul 2025 – Aug 2025 | Multi-year summer validation |
| `v5` | Jul 2023 – Dec 2025 | 30 months | Jan 2026 – Feb 2026 | Winter demand shift analysis |
| `v6` | Jul 2023 – Apr 2026 | 34 months | May 2026 – Jun 2026 | Spring ridership acceleration |
| **`v7`** | **Jul 2023 – Jun 2026** | **36 months** | **Jul 2026 – Aug 2026** | **Active Production Dataset (833K+ records)** |

All dataset configurations are centrally managed in [config.yaml](config.yaml).

---

## 5. DVC Hyperparameter Experimentation

BikeSense AI integrates DVC experiment tracking (`dvc exp run`) with MLflow logging, allowing reproducibility of hyperparameter sweeps directly from the terminal.

### Running an Automated Multi-Experiment Sweep

We provide a dedicated automation script that evaluates multiple hyperparameter configurations, displays comparative metrics, and automatically applies the winning model to the workspace:

```bash
python scripts/run_experiments_and_apply.py
```

### Running Individual DVC Experiments Manually

You can override any parameter in [configs/model.yaml](configs/model.yaml) without editing the file:

```bash
# Run experiment with overridden random seed and regularization alpha
dvc exp run -n exp_custom -S configs/model.yaml:tuning.random_state=42 -S configs/model.yaml:ridge.alpha=1.0

# Compare all historical experiments in a table
dvc exp show

# Apply the parameters and artifacts of an experiment to your working directory
dvc exp apply <EXPERIMENT_NAME>
```

---

## 6. Model Registry & Dynamic Promotion Gate

### Governance Aliases in MLflow Model Registry
* **`@champion`**: The active model served in production.
* **`@challenger`**: The newly trained candidate model undergoing promotion evaluation.
* **`@previous_champion`**: The displaced champion retained for instant zero-code rollback.

### Automated Promotion Gate Logic
A challenger is promoted to `@champion` only if it satisfies both criteria in [configs/gate.yaml](configs/gate.yaml):
1. **MAE Improvement:** Challenger holdout MAE must improve over the champion by $\ge 3.0\%$.
2. **Shortfall Guardrail:** Under-forecast shortfall must not worsen by $> 5.0\%$.

```bash
# Register winning candidate into registry as @challenger
python -m src.registry.register

# Evaluate promotion gate (compares @challenger vs @champion)
python -m src.registry.promote --auto-approve

# Instant rollback to a specific version if needed
python -m src.registry.rollback --to 1
```

---

## 7. Repository Layout

```
bikesense-ai/
├── app/                              # Production Serving Microservice
│   ├── main.py                       # FastAPI application & REST endpoints
│   ├── schemas.py                    # Pydantic request/response validation
│   └── static/
│       └── index.html                # Interactive Chart.js web dashboard
├── configs/                          # Experiment & Operational Configs
│   ├── features.yaml                 # Feature engineering definitions
│   ├── gate.yaml                     # Promotion gate thresholds & rules
│   ├── model.yaml                    # Hyperparameter distributions & model specs
│   └── monitoring.yaml               # Drift & alerting thresholds
├── data/                             # DVC-Tracked Data Partitions
│   ├── raw/                          # Ingested Citi Bike trip records
│   ├── external/                     # Cached Open-Meteo weather & holidays
│   └── processed/                    # Feature store parquet tables
├── docs/                             # Operational & Architectural Documentation
│   ├── running_servers_guide.md      # Guide for running MLflow + FastAPI
│   └── dvc_s3_guide.md               # S3 remote setup & DVC workflow guide
├── reports/                          # Generated Evaluation & Monitoring Reports
│   └── evaluation/
│       ├── latest_training_summary.json  # Latest experiment metrics
│       └── latest_evaluation.md          # Head-to-head model comparison report
├── scripts/                          # Automation & Utility Scripts
│   ├── run_experiments_and_apply.py  # DVC multi-experiment sweep runner
│   ├── registry_audit.py             # Lineage audit across all model versions
│   └── run_demo.sh                   # End-to-end local demonstration script
├── service/                          # Shared Service Utilities
│   └── app/
│       └── model_loader.py           # DynamicModelLoader (zero-downtime hot-reload)
├── src/                              # Core Source Code
│   ├── data/                         # Ingest, weather, holidays, and validation
│   ├── features/                     # build_features pipeline
│   ├── models/                       # train, evaluate, baseline, and metrics
│   ├── monitoring/                   # Evidently drift reports, alerts, rolling error
│   └── registry/                     # MLflow register, promote, and rollback
├── tests/                            # 38 Comprehensive Unit & Integration Tests
├── .github/workflows/                # GitHub Actions Automation
│   ├── ci.yml                        # Lint, type-check, tests, DVC dry-run
│   ├── cd.yml                        # Docker build, Amazon ECR push, EC2 deploy
│   └── retrain.yml                   # Weekly automated model retraining
├── config.yaml                       # Global pipeline parameters & dataset versions
├── dvc.yaml                          # 5-stage DVC pipeline definition
├── dvc.lock                          # Pinned content hashes per pipeline stage
├── Dockerfile                        # Production-ready, EC2-compatible Docker image
├── docker-compose.yml                # Multi-service stack (FastAPI + MLflow)
├── requirements.txt                  # Locked Python dependencies
└── Project_Description.md            # Comprehensive Technical Specification
```

---

## 8. Quickstart: Running Locally

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Git and DVC
* ~10 GB free disk space (for trip data archives)

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Emperor2004/bikesense-ai.git
cd bikesense-ai

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Reproduce Pipeline via DVC
```bash
# Pull cached artifacts from S3 or reproduce from scratch
dvc pull || dvc repro
```

### 3. Launch the Stack (Option: MLflow + FastAPI)

#### Terminal 1 — MLflow Tracking & Registry Server:
```bash
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```
*Access MLflow UI at:* [http://127.0.0.1:5000](http://127.0.0.1:5000)

#### Terminal 2 — FastAPI Serving & Interactive Web Dashboard:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Access Web Dashboard at:* [http://127.0.0.1:8000](http://127.0.0.1:8000)  
*Access Swagger API Docs at:* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 9. Running with Docker & Cloud Deployment

### Standalone Docker Container (Amazon EC2 Compatible)
```bash
# Build the self-contained production container
docker build -t bikesense-ai:latest .

# Run container mapping port 8000
docker run -d -p 8000:8000 --name bikesense-service bikesense-ai:latest

# Verify health
curl http://localhost:8000/health
```

### Multi-Container Stack (Docker Compose)
To run both the prediction service and the local MLflow tracking server together:
```bash
docker compose up --build -d
```

---

## 10. Continuous Integration & Continuous Deployment (CI/CD)

The project includes production-grade GitHub Actions workflows located in `.github/workflows/`:

1. **Continuous Integration (`ci.yml`):**
   * Triggered on every pull request and push to `main`.
   * Enforces code formatting (`ruff check`), type safety, and executes the entire 38-case test suite (`pytest tests/`).
   * Validates DVC pipeline integrity with dry-run verification.
2. **Continuous Deployment (`cd.yml`):**
   * Triggered on merges to `main`.
   * Builds the optimized Docker image and pushes it to **Amazon ECR**.
   * Deploys directly to target **Amazon EC2** instances via SSH, pulling the updated image and restarting the container with zero manual intervention.
3. **Automated Weekly Retraining (`retrain.yml`):**
   * Scheduled cron trigger every Sunday at 02:00 UTC (`0 2 * * 0`).
   * Fetches latest trip data, reproduces the DVC pipeline, evaluates candidate models, and triggers the promotion gate.

---

## 11. Monitoring & Drift Detection

BikeSense AI monitors incoming production inference requests using a delayed ground-truth architecture:

* **Delayed Actuals Join:** Predictions logged to `data/predictions/prediction_log.csv` are joined with ground-truth Citi Bike departures 24 hours later.
* **Evidently AI Drift Reports:** Evaluates statistical distribution shifts in meteorological variables (temperature, precipitation, wind) and temporal features using Kolmogorov-Smirnov and Wasserstein distance tests.
* **Alert Engine (`src/monitoring/alerts.py`):**
  * `ALERT-101`: Triggered when $> 30\%$ of input features exhibit distribution drift.
  * `ALERT-102`: Triggered when rolling 7-day MAE exceeds $1.25\times$ reference training MAE.
  * `ALERT-103`: Triggered when under-forecast shortfall surges, warning operators of imminent station stockouts.

---

## 12. Benchmark Results

Models evaluated on the 36-month (`v7`) holdout replay window (July – August 2026):

| Model Name | Version | Holdout MAE | RMSE | Shortfall (Units) | Promotion Outcome |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Seasonal-Naive Baseline** | — | 1.8410 | 2.6512 | 19,840.00 | Reference Benchmark |
| **LightGBM Shallow** | `v1` | 1.3864 | 2.0411 | 13,934.50 | Displaced Champion |
| **LightGBM Tuned (Trial 3)** | `v2` | 1.3992 | 2.0513 | 13,963.28 | Rejected (Insufficient gain) |
| **LightGBM Tuned (Trial 2)** | **`v3`** | **1.3153** | **2.0037** | **14,179.34** | 🏆 **PROMOTED TO @champion (+5.13% gain)** |

---

## 13. Testing & Quality Assurance

BikeSense AI includes a comprehensive test suite of **38 automated tests**:

```bash
# Run tests with short traceback output
pytest tests/ -v --tb=short

# Run code linter
ruff check src/ tests/ app/
```

| Module | Test File | Test Count | Scope |
| :--- | :--- | :---: | :--- |
| **Data Validation** | `tests/data/` | 7 | Schema assertions, date validity, non-negative guards |
| **Feature Engineering** | `tests/features/` | 5 | Temporal features, lag leakage prevention, rolling means |
| **Model Evaluation** | `tests/models/` | 9 | MAE/RMSE/shortfall metrics, time-ordered split integrity |
| **Registry & Gates** | `tests/registry/` | 6 | Gate approval/rejection, shortfall guardrails, rollback |
| **Serving Endpoints** | `tests/service/` | 5 | Health, forecast, model-info endpoints, prediction logging |
| **Monitoring & Alerts**| `tests/monitoring/`| 6 | Evidently drift checks, rolling error, alert triggers |

---

## 14. Team & License

### Authors
* **Om Narayan Pandit** — *Data Ingestion, DVC Pipelines, Feature Engineering, CI/CD, AWS Deployment*
* **Pranjali Vishwakarma** — *Modeling, Evaluation Metrics, MLflow Model Registry, Promotion Gate*
* **Rajat Dwivedi** — *FastAPI Serving, Web Dashboard, Containerization, Evidently AI Monitoring*

### License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.  
Citi Bike trip data is subject to the [NYCBS Data Use Policy](https://citibikenyc.com/system-data). Historical weather data provided by [Open-Meteo](https://open-meteo.com) under CC BY 4.0.
