# 🚲 BikeSense AI: Production MLOps Demand Forecasting Microservice

[![Python 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![DVC 3.x](https://img.shields.io/badge/DVC-3.48+-945DD6.svg?logo=dvc&logoColor=white)](https://dvc.org)
[![MLflow 2.12+](https://img.shields.io/badge/MLflow-2.12+-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-29.x-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20ECR%20%7C%20EC2-FF9900.svg?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![CI/CD Status](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD%20Passing-success.svg?logo=github-actions&logoColor=white)](https://github.com/Emperor2004/bikesense-ai/actions)
[![Test Suite](https://img.shields.io/badge/Tests-38%20Passing-brightgreen.svg?logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, end-to-end Machine Learning Operations (MLOps) platform that delivers hourly station-level bicycle departure forecasts for Citi Bike (New York / Jersey City) across **38 continuous months (July 2023 – August 2026)**. 

The system features automated DVC pipeline reproducibility with AWS S3 remote storage, multi-model hyperparameter experimentation, MLflow Model Registry governance with zero-downtime hot-reloading, an interactive operational dashboard with automated rebalancing dispatch recommendations, and robust GitHub Actions CI/CD workflows targeting Amazon ECR and EC2.

> 📖 **Deep-Dive Engineering Documentation:**  
> For in-depth mathematical formulations, feature engineering definitions, loss functions, and architectural decisions, see [Project_Description.md](Project_Description.md).

---

## 📌 Table of Contents

1. [Key Capabilities & Business Value](#1-key-capabilities--business-value)
2. [Interactive Operations Dashboard & REST Endpoints](#2-interactive-operations-dashboard--rest-endpoints)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [Dataset Versions & Progressive Replay Windows](#4-dataset-versions--progressive-replay-windows)
5. [DVC Experiment Sweeps & Model Training](#5-dvc-experiment-sweeps--model-training)
6. [Model Governance & Automated Promotion Gate](#6-model-governance--automated-promotion-gate)
7. [Repository Layout](#7-repository-layout)
8. [Quickstart: Running Locally](#8-quickstart-running-locally)
9. [Docker Deployment & Production Serving](#9-docker-deployment--production-serving)
10. [Continuous Integration & Continuous Deployment (CI/CD)](#10-continuous-integration--continuous-deployment-cicd)
11. [Monitoring, Drift Detection & Alerting](#11-monitoring-drift-detection--alerting)
12. [Benchmark Evaluation Results](#12-benchmark-evaluation-results)
13. [Testing & Quality Assurance](#13-testing--quality-assurance)
14. [Contributors & License](#14-contributors--license)

---

## 1. Key Capabilities & Business Value

Bike-share systems face continuous operational friction: **starvation events** (empty stations) strand commuters and forfeit revenue, while **overflow events** (full stations) prevent riders from docking. Dispatching rebalancing trucks requires accurate 24-hour departure forecasts to stage fleet capacity before morning and evening rush hours.

| Capability | Problem Solved | Implementation |
| :--- | :--- | :--- |
| **Data Versioning & Lineage** | Non-reproducible feature stores and data drift across 38 months | DVC backed by Amazon S3 (`s3://om-dvc-bucket`) |
| **Hyperparameter Sweeps** | Manual, uncoordinated model tuning | `dvc exp run` + scikit-learn `ParameterSampler` |
| **Model Registry Governance** | Uncontrolled production deployments and lack of provenance | MLflow Model Registry with `@champion`, `@challenger`, and `@previous_champion` |
| **Zero-Downtime Serving** | Service interruptions and dropped HTTP traffic during model promotion | `DynamicModelLoader` with in-memory caching and background alias resolution |
| **Lightweight Serving Image** | Heavy offline packages (training/DVC/Evidently) bloating production containers | Optimized `requirements-serve.txt` resulting in a ~387 MB container image |
| **Operational Dashboard** | Dispatchers need clear, real-time predictions and actionable guidance | Modern Glassmorphism Dashboard served at `/` with automated rebalancing alerts |
| **Hardened CI/CD Pipelines** | Stale images and regression bugs escaping to production | GitHub Actions: Linting, 38 unit tests, Docker health checks, ECR push, and EC2 auto-deploy |

---

## 2. Interactive Operations Dashboard & REST Endpoints

The FastAPI microservice in [`app/main.py`](app/main.py) serves an interactive operational dashboard directly from the root `/` endpoint alongside documented RESTful API endpoints.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  🚲 BikeSense AI — Citi Bike Demand Forecasting Dashboard                              │
│  Serving Champion: Version 3  |  Dataset: v7 (38 Mos)  |  Lineage: [0ca17b7...]         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Station: [ Grove St PATH (JC115) ▼ ]      Date: [ 2026-08-15 ]      [Refresh Forecast] │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  [ Total Departures: 412 ]   [ Morning Peak: 08:00 (34 bikes) ]   [ Evening: 18:00 ]   │
│                                                                                        │
│  Rebalancing Dispatch Recommendation:                                                  │
│  ⚠️ High Demand Alert: Stage at least 41 bicycles before morning peak commute (08:00).   │
│                                                                                        │
│  Hourly Departure Trajectory (00:00 - 23:00):                                          │
│   40 ┤                      ╭──╮                                                       │
│   30 ┤        ╭─╮           │  │                                                       │
│   20 ┤        │ │   ╭───╮   │  │       ─── LightGBM Champion Model                     │
│   10 ┤    ╭───╯ ╰───╯   ╰───╯  ╰───╮   - - Seasonal-Naive Baseline (168h)              │
│    0 ┴────┴────────────────────────┴──► Hour                                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### API Endpoints Overview

| Endpoint | Method | Request / Query | Response Model | Description |
| :--- | :---: | :--- | :--- | :--- |
| `/` | `GET` | — | HTML5 / JavaScript | Interactive forecasting dashboard with live Chart.js visualizations |
| `/health` | `GET` | — | `HealthResponse` | Service health status, server uptime, and loaded `@champion` version |
| `/model-info` | `GET` | — | `ModelInfoResponse` | Lineage tags (Git commit SHA, DVC data hash, run ID, load timestamp) |
| `/stations` | `GET` | — | `list[StationItem]` | All 30 monitored Citi Bike stations with human-readable names and hub flags |
| `/forecast` | `GET` | `station_id`, `date` | `ForecastResponse` | 24-hour departure forecast, peak hour metrics, baseline, and rebalancing alerts |
| `/predict` | `POST` | `PredictRequest` (JSON) | `PredictResponse` | Direct batch scoring endpoint accepting raw feature records |
| `/docs` | `GET` | — | OpenAPI / Swagger UI | Interactive API documentation and testing console |

---

## 3. End-to-End System Architecture

```
[ Citi Bike Trip CSV Archives ] ──┐
[ Open-Meteo Weather API (ERA5) ] ──┼─► Ingestion & QC (src/data/ingest.py) ──► data/raw/departures_hourly.parquet
[ US Federal Holidays Calendar  ] ──┘                 │
                                                      ▼
                                       Feature Store (src/features/build_features.py)
                                       (24h/168h Lags, 7d Rolling Means, Weather, Holidays)
                                                      │
                                                      ▼
                                       Multi-Model Training (src/models/train.py)
                                       ├── Baseline: 168-hour seasonal persistence
                                       ├── Linear: L2 Ridge Regression
                                       └── Boosted Trees: LightGBM (Tuned hyperparameter trials)
                                                      │
                                                      ▼
                                       MLflow Tracking & Model Registry
                                       (Lineage tags: Git Commit SHA + DVC Data Hash + Artifacts)
                                                      │
                                                      ▼
                                       Automated Promotion Gate (src/registry/promote.py)
                                       (MAE Gain >= 3.0%, Shortfall Surge <= 5.0%)
                                                      │
                                                      ▼
                                       FastAPI Serving & Web Dashboard (app/main.py)
                                       (Zero-Downtime Dynamic Model Hot-Reloading)
                                                      │
                                                      ▼
                                       CI/CD & Monitoring (GitHub Actions)
                                       (Automated AWS ECR/EC2 Deployment + Evidently AI Drift Reports)
```

---

## 4. Dataset Versions & Progressive Replay Windows

The pipeline versions **38 continuous months of data (July 2023 – August 2026)**, modeling ridership shifts, extreme weather patterns, seasonal demand drops, and station capacity changes:

| Version | Training Range | Training Span | Holdout Replay Window | Purpose |
| :---: | :---: | :---: | :---: | :--- |
| `v1` | Jul 2023 – Feb 2024 | 8 months | Mar 2024 – Apr 2024 | Baseline model structure and pipeline validation |
| `v2` | Jul 2023 – Apr 2024 | 10 months | May 2024 – Jun 2024 | First seasonal expansion (Spring surge) |
| `v3` | Jul 2023 – Jun 2024 | 12 months | Jul 2024 – Aug 2024 | Complete annual calendar cycle |
| `v4` | Jul 2023 – Jun 2025 | 24 months | Jul 2025 – Aug 2025 | Multi-year summer comparison |
| `v5` | Jul 2023 – Dec 2025 | 30 months | Jan 2026 – Feb 2026 | Deep winter demand shift and freezing precipitation |
| `v6` | Jul 2023 – Apr 2026 | 34 months | May 2026 – Jun 2026 | Spring commute acceleration |
| **`v7`** | **Jul 2023 – Jun 2026** | **36 months** | **Jul 2026 – Aug 2026** | **Active Production Dataset (833K+ hourly records)** |

Dataset definitions, window dates, and station lists are centrally configured in [`config.yaml`](config.yaml).

---

## 5. DVC Experiment Sweeps & Model Training

BikeSense AI orchestrates hyperparameter optimization using DVC (`dvc exp run`) combined with MLflow tracking.

### Automated Multi-Trial Sweep
Run an automated multi-configuration hyperparameter search that compares candidates, logs metrics to MLflow, and automatically applies the top performer:

```bash
python scripts/run_experiments_and_apply.py
```

### Manual Experiment Runs via DVC
Override parameters in [`configs/model.yaml`](configs/model.yaml) without modifying files:

```bash
# Run an experiment trial with custom tree depth and learning rate
dvc exp run -n exp_trial_custom -S configs/model.yaml:lightgbm.params.max_depth=7 -S configs/model.yaml:lightgbm.params.learning_rate=0.03

# Compare experiment results in the terminal
dvc exp show

# Apply winning parameters to the active workspace
dvc exp apply <EXPERIMENT_NAME>
```

---

## 6. Model Governance & Automated Promotion Gate

### Governance Aliases in MLflow Model Registry
* **`@champion`**: The active model currently loaded and serving production traffic.
* **`@challenger`**: The newly trained candidate model undergoing holdout verification.
* **`@previous_champion`**: The displaced champion retained for zero-downtime rollback.

### Quantitative Promotion Criteria ([`configs/gate.yaml`](configs/gate.yaml))
A challenger is promoted to `@champion` only when passing both thresholds:
1. **MAE Improvement:** Holdout MAE must improve over the champion by $\ge 3.0\%$.
2. **Shortfall Guardrail:** Under-forecast shortfall must not increase by $> 5.0\%$ (protects against station starvation).

```bash
# Register winning training run as @challenger
python -m src.registry.register

# Evaluate promotion gate
python -m src.registry.promote --auto-approve

# Immediate zero-downtime rollback to a previous version if required
python -m src.registry.rollback --to 1
```

---

## 7. Repository Layout

```
bikesense-ai/
├── app/                                # FastAPI Serving Application
│   ├── main.py                         # FastAPI routes, endpoints & dashboard server
│   └── static/
│       └── index.html                  # Interactive Glassmorphism Chart.js dashboard
├── configs/                            # Central Configuration Files
│   ├── features.yaml                   # Feature store schema and lag definitions
│   ├── gate.yaml                       # Promotion gate thresholds & rules
│   ├── model.yaml                      # Model hyperparameter spaces
│   └── monitoring.yaml                 # Evidently AI drift & alert thresholds
├── data/                               # DVC-Tracked Data Partitions
│   ├── raw/                            # departures_hourly.parquet
│   ├── external/                       # weather/ and holidays.parquet
│   └── processed/                      # features.parquet & station metadata
├── docs/                               # Operational & Architecture Guides
│   ├── running_servers_guide.md        # Server execution guide
│   └── dvc_s3_guide.md                 # S3 remote setup & DVC workflow guide
├── reports/                            # Evaluation & Monitoring Artifacts
│   └── evaluation/                     # Training summaries and evaluation reports
├── scripts/                            # Automation & Experiment Scripts
│   ├── run_experiments_and_apply.py    # Automated DVC experiment sweep runner
│   ├── registry_audit.py               # MLflow registry lineage inspector
│   └── run_demo.sh                     # End-to-end local demo script
├── service/                            # Runtime Serving Modules
│   ├── init_stub_mlflow.py             # Schema initializer for container builds
│   └── app/
│       ├── model_loader.py             # DynamicModelLoader (zero-downtime hot-reload)
│       ├── prediction_log.py           # Real-time inference request logger
│       └── schemas.py                  # Pydantic request & response models
├── src/                                # Core Pipeline Packages
│   ├── data/                           # Data ingest, weather extraction, holidays, QC
│   ├── features/                       # Time-series feature engineering
│   ├── models/                         # Training, baseline heuristics, and metrics
│   ├── monitoring/                     # Evidently drift detection & alert engine
│   └── registry/                       # Registration, promotion gate & rollback
├── tests/                              # 38 Automated Unit & Integration Tests
├── .github/workflows/                  # Production GitHub Actions
│   ├── ci.yml                          # Lint, pytest, coverage, Docker build & healthcheck
│   ├── cd.yml                          # Build & push to AWS ECR, deploy to Amazon EC2
│   └── retrain.yml                     # Scheduled weekly S3 DVC pull & retrain
├── Dockerfile                          # Optimized, multi-platform production Dockerfile
├── docker-compose.yml                  # Local development stack (FastAPI + MLflow)
├── requirements.txt                    # Full MLOps training & pipeline dependencies
├── requirements-serve.txt              # Lightweight runtime serving dependencies (~387 MB image)
├── config.yaml                         # Global dataset versions & station parameters
├── dvc.yaml                            # 5-stage DVC pipeline definition
└── Project_Description.md              # In-depth technical specification & design
```

---

## 8. Quickstart: Running Locally

### Prerequisites
* Python 3.10 or 3.11
* Git & Docker (optional for containerized execution)
* Minimum 8 GB RAM

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/Emperor2004/bikesense-ai.git
cd bikesense-ai

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Pull Data & Models via DVC

```bash
# Pull versioned features and models from AWS S3 remote
dvc pull

# Or reproduce the full pipeline from raw data
dvc repro
```

### 3. Launch Services

#### Launch the Web Dashboard & API (Port 8000)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
* **Interactive Dashboard:** [http://localhost:8000/](http://localhost:8000/)  
* **Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)  
* **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

#### Launch the MLflow Tracking UI (Port 5000)
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --host 0.0.0.0
```
* **MLflow Tracking & Registry UI:** [http://localhost:5000/](http://localhost:5000/)

---

## 9. Docker Deployment & Production Serving

The project uses an optimized [`requirements-serve.txt`](requirements-serve.txt) and [`Dockerfile`](Dockerfile) to bundle only inference dependencies, resulting in a fast, lightweight container image.

### Building & Running with Docker

```bash
# 1. Build the production Docker image
docker build -t bikesense-forecast:latest .

# 2. Run the container
docker run -d -p 8000:8000 --name bikesense-api --restart unless-stopped bikesense-forecast:latest

# 3. Verify container health
curl http://localhost:8000/health

# 4. View container logs
docker logs -f bikesense-api

# 5. Stop and clean up
docker stop bikesense-api && docker rm bikesense-api
```

### Running with Docker Compose
To run both the prediction service and MLflow tracking server together:
```bash
docker compose up -d --build
```

---

## 10. Continuous Integration & Continuous Deployment (CI/CD)

The repository includes three production-ready GitHub Actions workflows:

### 1. Continuous Integration (`ci.yml`)
Runs on every pull request and push to `main`:
* **Code Quality & Linting:** Strict Ruff checks across `src/`, `app/`, `service/`, and `tests/`.
* **Automated Test Suite:** Executes all 38 pytest unit and integration tests with XML coverage reporting.
* **Container Build & Health Verification:** Generates stub MLflow artifacts via [`service/init_stub_mlflow.py`](service/init_stub_mlflow.py), builds the Docker container, boots it on port 8000, and verifies `/health` and dashboard `/` responses.

### 2. Continuous Deployment (`cd.yml`)
Triggers automatically on merges to `main`:
* **AWS Authentication:** Connects via `aws-actions/configure-aws-credentials` using GitHub Secrets.
* **Amazon ECR:** Logs into AWS ECR, verifies repository availability, builds the production container, and pushes both `:${{ github.sha }}` and `:latest` tags.
* **Amazon EC2 Deployment:** Securely connects to target EC2 instances over SSH, pulls the updated container image from ECR, and replaces the running service with zero downtime.

### 3. Automated Weekly Retraining (`retrain.yml`)
* **Scheduled Cron:** Executes weekly on Sundays at midnight UTC (`0 0 * * 0`).
* **Workflow:** Pulls cached data from S3, executes `dvc repro`, registers candidate models, evaluates the promotion gate with `--auto-approve`, and pushes updated artifacts to S3.

---

## 11. Monitoring, Drift Detection & Alerting

* **Delayed Ground-Truth Join (`src/monitoring/join_actuals.py`):** Automatically joins real-time predictions logged to `data/predictions/prediction_log.csv` with true Citi Bike ride departures after a 24-hour lag.
* **Evidently AI Statistical Drift (`src/monitoring/drift_report.py`):** Computes Kolmogorov-Smirnov and Wasserstein distance tests across weather and lag feature distributions.
* **Alert Engine (`src/monitoring/alerts.py`):**
  * `ALERT-101`: Triggered when $> 30\%$ of features exhibit distribution drift.
  * `ALERT-102`: Triggered when rolling 7-day MAE exceeds $1.25\times$ reference training MAE.
  * `ALERT-103`: Triggered when under-forecast shortfall surges, warning operators of imminent station stockouts.

---

## 12. Benchmark Evaluation Results

All candidate models evaluated on the 36-month (`v7`) holdout replay window (July – August 2026):

| Model Candidate | Model Architecture | Holdout MAE | RMSE | Shortfall (Bikes) | Gate Decision |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline** | Seasonal-Naive Heuristic (168h) | 1.8410 | 2.6512 | 19,840.00 | Reference Benchmark |
| **Version 1** | LightGBM Shallow (`lgbm_shallow`) | 1.3864 | 2.0411 | 13,934.50 | Displaced Champion |
| **Version 2** | LightGBM Tuned Trial 3 (`lgbm_tune_trial_3`) | 1.3992 | 2.0513 | 13,963.28 | Rejected (Insufficient gain) |
| **Version 3** | **LightGBM Tuned Trial 2 (`lgbm_tune_trial_2`)** | **1.3153** | **2.0037** | **14,179.34** | 🏆 **PROMOTED TO @champion (+5.13% gain)** |

---

## 13. Testing & Quality Assurance

BikeSense AI maintains a comprehensive suite of **38 automated unit and integration tests**:

```bash
# Run complete test suite with concise output
pytest tests/ -v --tb=short

# Run test coverage analysis
pytest tests/ --cov=src --cov=app --cov-report=term-missing

# Run code style and linting
ruff check src/ app/ service/ tests/
```

| Test Module | Location | Tests | Verification Scope |
| :--- | :--- | :---: | :--- |
| **Data Ingestion & QC** | `tests/data/` | 7 | Schema enforcement, date boundaries, zero-fill completeness |
| **Feature Engineering** | `tests/features/` | 5 | Temporal features, lag calculation, temporal leakage prevention |
| **Model Evaluation** | `tests/models/` | 9 | Metric correctness (MAE, RMSE, Shortfall), train/test temporal split integrity |
| **Registry & Gates** | `tests/registry/` | 6 | Gate thresholds, auto-approval, shortfall guardrails, rollback mechanics |
| **FastAPI Services** | `tests/service/` | 5 | `/health`, `/forecast`, `/model-info`, `/predict`, prediction logging |
| **Monitoring & Drift** | `tests/monitoring/` | 6 | Evidently report generation, rolling error, alert trigger logic |

---

## 14. Contributors & License

### Authors
* **Om Narayan Pandit** — *Data Ingestion, DVC Pipelines, Feature Engineering, CI/CD, AWS Deployment*
* **Pranjali Vishwakarma** — *Modeling, Evaluation Metrics, MLflow Model Registry, Promotion Gate*
* **Rajat Dwivedi** — *FastAPI Serving, Web Dashboard, Containerization, Evidently AI Monitoring*

### License & Data Credits
* This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
* Citi Bike trip data is published by NYCBS/Lyft under the [Citi Bike Data Agreement](https://citibikenyc.com/system-data).
* Meteorological data provided by [Open-Meteo](https://open-meteo.com) under CC BY 4.0.
