# BikeSense AI — Comprehensive Engineering Specification & Architecture Manual

## 1. Executive Summary & Problem Context

**BikeSense AI** is a production-grade Machine Learning Operations (MLOps) platform engineered to solve the complex spatio-temporal rebalancing challenge in the Citi Bike network across New York City and Jersey City.

Modern urban bike-share networks operate under intense asymmetric demand pressures:
* **Morning Commute Surges:** Residential stations deplete rapidly as commuters take bikes toward transit hubs (e.g., Grove St PATH, Exchange Place, Hoboken Terminal), leaving residential docks completely empty (**station starvation**).
* **Evening Commute Inversions:** Riders return bikes from commercial centers back to residential neighborhoods, quickly filling all available docks (**station overflow**) and preventing incoming riders from returning bicycles.
* **Severe Financial & Operational Penalties:** When stations starve, operators lose trip revenues and commuters face stranded trips. When stations overflow, riders are forced to search for alternative docks, resulting in route abandonment and regulatory fines from municipal transport authorities.

### The Role of Predictive MLOps in Fleet Operations
Truck-based fleet rebalancing cannot react in real time to sudden spikes; dispatchers must preposition bicycles during low-traffic overnight hours (00:00–05:00) and midday lulls (11:00–14:00). Effective rebalancing requires **accurate, 24-hour lookahead departure forecasts at the individual station level**.

However, static machine learning models degrade rapidly in production due to:
1. **Covariate Shift:** Fluctuating weather patterns (severe rain, snowstorms, sudden heatwaves).
2. **Calendar Discontinuities:** Federal and state holidays altering commuter flow.
3. **Concept Drift:** Progressive network expansion, post-pandemic transit shifts, and urban infrastructure modifications.

BikeSense AI delivers an end-to-end, automated, drift-aware MLOps lifecycle incorporating DVC data versioning backed by Amazon S3, multi-model experimentation with MLflow, zero-downtime dynamic model serving, an interactive operations dashboard with dispatch recommendations, automated promotion gates, and robust CI/CD workflows targeting Amazon ECR and EC2.

---

## 2. Mathematical Formulation & Optimization Objectives

### 2.1 The Forecasting Task
* **Spatial Dimension:** A set of $S = 30$ top-volume stations, denoted $s \in \{1, 2, \dots, S\}$.
* **Temporal Dimension:** Hourly discrete time steps $t \in \mathcal{T}$.
* **Target Variable ($y_{s,t}$):** The total number of bicycle departures originating from station $s$ during the hourly interval $[t, t + 1\text{ hour})$.
* **Forecast Horizon:** A vector of 24 future departure values issued daily at 00:00 UTC:
  $$\hat{\mathbf{Y}}_{s, T} = \left[ \hat{y}_{s, T+1}, \hat{y}_{s, T+2}, \dots, \hat{y}_{s, T+24} \right]$$
* **Information Boundary Constraint:** For any forecast issued at timestamp $T$, the feature vector $\mathbf{x}_{s, t}$ may strictly utilize only information available prior to $T$:
  $$\mathcal{I}_T = \{ y_{s, \tau}, \mathbf{w}_{\tau}, \mathbf{c}_{\tau} \mid \tau < T \}$$
  where $\mathbf{w}$ represents meteorological observations and $\mathbf{c}$ represents calendar metadata.

### 2.2 Objective Functions & Evaluation Metrics

To balance statistical precision with operational utility, models are evaluated across three complementary metrics:

#### 1. Mean Absolute Error (Primary Optimization Metric)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left| y_i - \hat{y}_i \right|$$
MAE provides a linear, interpretable measure of departure deviation per station-hour, treating positive and negative deviations equally without disproportionately penalizing standard operational noise.

#### 2. Root Mean Squared Error (Outlier & Peak Sensitivity)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2}$$
By squaring error deviations before averaging, RMSE penalizes large errors heavily, serving as a vital diagnostic for peak commute hours where underestimating high demand causes massive service disruptions.

#### 3. Under-Forecast Shortfall (Asymmetric Operational Risk Proxy)
$$\text{Shortfall} = \sum_{i=1}^N \max\left(0, y_i - \hat{y}_i\right)$$
In bike-share rebalancing, errors are non-symmetric:
* **Over-predicting demand ($\hat{y} > y$):** Results in staging excess bicycles at a station, consuming dock buffer.
* **Under-predicting demand ($\hat{y} < y$):** Leaves a station under-resourced, resulting in empty docks, stranded commuters, and unrecoverable revenue loss.

Shortfall directly quantifies the cumulative unpredicted demand units across the holdout window, proxying lost ride opportunities.

---

## 3. System Architecture & High-Level Pipeline

The end-to-end architecture is organized into modular, decoupled subsystems governed by DVC and MLflow:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA INGESTION & VERSIONING                               │
│  - Citi Bike S3 Monthly Archives (Parquet/CSV)                                         │
│  - Open-Meteo Historical Weather API (ERA5 Reanalysis)                                 │
│  - US Federal & State Holiday Calendars (Python holidays library)                       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA QUALITY CONTROL & QC                                 │
│  - Schema enforcement, timestamp normalization to UTC                                  │
│  - Non-negative departures validation                                                  │
│  - Cartesian product grid expansion (30 stations x 8,760 hours) + Zero-filling         │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             FEATURE STORE (DVC VERSIONED)                              │
│  - Temporal: hour [0-23], day_of_week [0-6], month [1-12], is_holiday [0/1]            │
│  - Weather: temperature_2m (°C), precipitation (mm), wind_speed_10m (km/h)             │
│  - Autoregressive: lag_24h (yesterday), lag_168h (last week)                           │
│  - Rolling Statistics: rolling_7d_same_hour_mean (7-day historical window)             │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               MODEL TRAINING & HYPER-TUNING                            │
│  - Baseline: Seasonal-Naive 168-hour persistence heuristic                             │
│  - Linear: L2 Regularized Ridge Regression                                             │
│  - Gradient Boosters: LightGBM Regressors (Shallow, Deep, Regularized, Tuned)          │
│  - Exploration: sklearn ParameterSampler + DVC Experiment Sweeps (`dvc exp run`)       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           MLFLOW EXPERIMENT TRACKING & REGISTRY                        │
│  - Run parameters, holdout metrics, model artifacts logged with SQLite backend          │
│  - Immutable Lineage Tags: Git commit SHA, DVC data hash, dataset version, run ID       │
│  - Aliases: @champion (Production), @challenger (Candidate), @previous_champion        │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              AUTOMATED PROMOTION GATE                                  │
│  - Quantitative Thresholds: MAE Improvement >= 3.0%, Shortfall Surge <= 5.0%           │
│  - Non-interactive auto-approval mode for automated CI/CD runs                         │
│  - Instant zero-downtime rollback mechanism (`src.registry.rollback`)                 │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                            PRODUCTION SERVING MICROSERVICE                             │
│  - FastAPI REST API: `/health`, `/model-info`, `/stations`, `/forecast`, `/predict`     │
│  - DynamicModelLoader: Thread-safe in-memory caching & zero-downtime hot-reloading     │
│  - Interactive Glassmorphism Dashboard with Chart.js & dispatch recommendation alerts │
│  - Optimized Docker Container: ~387 MB image using requirements-serve.txt              │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             MONITORING & CONTINUOUS DELIVERY                           │
│  - 24-Hour Delayed Actuals Join (`data/predictions/prediction_log.csv`)               │
│  - Evidently AI Covariate & Prediction Drift Reports (Kolmogorov-Smirnov & Wasserstein)│
│  - Multi-tier Alert Engine: Distribution drift, rolling error, and shortfall surge     │
│  - GitHub Actions CI/CD: Automated linting, 38 tests, Docker verify, AWS ECR/EC2 deploy│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Engineering & Feature Store Design

### 4.1 Invariant Top Station Selection
To prevent spatial non-stationarity caused by temporary pop-up stations or decommissioning, the station pool is anchored to the **top 30 highest-volume stations** ranked by departure activity during the initial reference period (`v1`). 

* Stored in: [`data/processed/station_list.json`](data/processed/station_list.json)
* Station Metadata: Mapped with human-readable names and transit hub tags in [`data/processed/station_names.json`](data/processed/station_names.json)
* Transit Hubs: Major multimodal connections such as Grove St PATH (`JC115`), Exchange Place (`JC116`), Newport PATH (`JC066`), and Hoboken Terminal (`HB101`/`HB102`).

### 4.2 Cartesian Product Grid & Zero-Filling
A common trap in mobility time-series modeling is "censored zeros": hours where no ride departures occurred are omitted from raw trip logs. Failing to include these leads to positive bias during overnight and severe storm periods.

BikeSense AI builds a complete Cartesian product:
$$\mathcal{G} = \mathcal{S}_{30} \times \mathcal{T}_{\text{hourly}}$$
A full left outer join merges aggregated trip departures onto this grid, explicitly filling unobserved intervals with $y_{s,t} = 0$.

### 4.3 Feature Definitions & Leakage Safeguards

| Feature Column | Data Type | Physical / Mathematical Interpretation |
| :--- | :---: | :--- |
| `hour` | Integer [0–23] | Captures diurnal commute rhythms (morning peak ~08:00, evening peak ~18:00) |
| `day_of_week` | Integer [0–6] | Weekly periodicity (weekday business commutes vs. weekend recreational riding) |
| `month` | Integer [1–12] | Annual macro-seasonality (summer peak volume vs. winter contraction) |
| `is_holiday` | Binary {0, 1} | Flag indicating whether the day is an official US Federal holiday |
| `temperature_2m` | Float (°C) | Direct comfort driver: higher temperatures correlate with increased departures up to 32°C |
| `precipitation` | Float (mm) | Immediate volume suppressant: rain or snow sharply reduces casual and commuter trips |
| `wind_speed_10m` | Float (km/h) | High winds deter riding and lower average trip velocities |
| `lag_24h` | Float | Observed departures at the exact same hour on the previous calendar day |
| `lag_168h` | Float | Observed departures at the exact same hour one week prior (7 days × 24 hours) |
| `rolling_7d_same_hour_mean` | Float | Moving average of departure volume at the exact same hour across the preceding 7 days |

**Leakage Prevention:** All lag indicators and rolling statistics are calculated strictly over historical intervals. Features for hour $t$ utilize observations up to $t - 24$ hours, ensuring zero temporal leakage into the 24-hour evaluation horizon.

---

## 5. Dataset Versioning Lifecycle & 38-Month Rolling Windows

The pipeline versions **38 continuous months of data (July 2023 through August 2026)**. The data lifecycle uses expanding historical training horizons evaluated on forward-looking, out-of-sample 2-month holdout replay windows:

```
v1: [ Jul 2023 - Feb 2024 ] ──► Holdout: [ Mar 2024 - Apr 2024 ]  (8-Month Train)
v2: [ Jul 2023 - Apr 2024 ] ──► Holdout: [ May 2024 - Jun 2024 ]  (10-Month Train)
v3: [ Jul 2023 - Jun 2024 ] ──► Holdout: [ Jul 2024 - Aug 2024 ]  (12-Month Train)
v4: [ Jul 2023 - Jun 2025 ] ──► Holdout: [ Jul 2025 - Aug 2025 ]  (24-Month Train)
v5: [ Jul 2023 - Dec 2025 ] ──► Holdout: [ Jan 2026 - Feb 2026 ]  (30-Month Train)
v6: [ Jul 2023 - Apr 2026 ] ──► Holdout: [ May 2026 - Jun 2026 ]  (34-Month Train)
v7: [ Jul 2023 - Jun 2026 ] ──► Holdout: [ Jul 2026 - Aug 2026 ]  (36-Month Train — Production)
```

### Detailed Dataset Partitioning Table

| Version Tag | Training Start | Training End | Training Span | Holdout Start | Holdout End | Total Records | Operational Role |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `v1` | 2023-07-01 | 2024-02-29 | 8 months | 2024-03-01 | 2024-04-30 | ~175,000 | Initial baseline architecture & pipeline testing |
| `v2` | 2023-07-01 | 2024-04-30 | 10 months | 2024-05-01 | 2024-06-30 | ~219,000 | Spring seasonal transition evaluation |
| `v3` | 2023-07-01 | 2024-06-30 | 12 months | 2024-07-01 | 2024-08-31 | ~263,000 | Full 1-year annual cycle capture |
| `v4` | 2023-07-01 | 2025-06-30 | 24 months | 2025-07-01 | 2025-08-31 | ~526,000 | Multi-year summer comparison |
| `v5` | 2023-07-01 | 2025-12-31 | 30 months | 2026-01-01 | 2026-02-28 | ~657,000 | Deep winter freeze & ridership contraction |
| `v6` | 2023-07-01 | 2026-04-30 | 34 months | 2026-05-01 | 2026-06-30 | ~745,000 | Post-expansion spring ridership surge |
| **`v7`** | **2023-07-01** | **2026-06-30** | **36 months** | **2026-07-01** | **2026-08-31** | **833,040** | **Active Production Model Training Set** |

---

## 6. Model Suite, Hyperparameter Tuning & Experimentation

### 6.1 Candidate Model Implementations

1. **Seasonal-Naive Persistence Heuristic (`src/models/baseline.py`):**
   $$\hat{y}_{s, t} = y_{s, t - 168}$$
   Leverages the strong 7-day cyclical nature of commuter travel. Serves as the primary operational benchmark that every machine learning candidate must beat.

2. **L2 Regularized Ridge Regression:**
   $$\min_{\mathbf{w}} \left\{ \frac{1}{2N} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2 \right\}$$
   Trained with standardized continuous weather features and one-hot encoded categorical temporal columns. Serves as the linear baseline.

3. **LightGBM Gradient Boosted Decision Trees:**
   LightGBM was selected as the champion architecture for its histogram-based splitting efficiency, native handling of skewed count distributions, and fast training speed across 800K+ records.

### 6.2 Hyperparameter Distribution Space ([`configs/model.yaml`](configs/model.yaml))
* `n_estimators`: `[100, 150, 200, 250]`
* `learning_rate`: `[0.02, 0.04, 0.06, 0.08]`
* `num_leaves`: `[31, 45, 63, 80]`
* `max_depth`: `[5, 6, 7, 8]`
* `reg_alpha` (L1 Regularization): `[0.0, 0.1, 0.5]`
* `reg_lambda` (L2 Regularization): `[0.0, 0.1, 0.5]`
* `subsample`: `[0.8, 1.0]`
* `colsample_bytree`: `[0.8, 1.0]`

### 6.3 Automated DVC Experiment Sweeps ([`scripts/run_experiments_and_apply.py`](scripts/run_experiments_and_apply.py))
Hyperparameter exploration is coordinated directly through DVC:
```bash
python scripts/run_experiments_and_apply.py
```
The script samples configurations via `sklearn.model_selection.ParameterSampler`, runs `dvc exp run -S <params>`, records all runs in MLflow, extracts holdout validation metrics, displays a comparative terminal table, and applies the optimal configuration to the workspace via `dvc exp apply`.

---

## 7. Model Governance & Production Promotion Gate

### 7.1 Provenance & Immutable Lineage Tags
Every registered model version in the MLflow Model Registry is stamped with immutable metadata tags:
* `git_commit`: Exact Git commit SHA (e.g. `0ca17b7...`).
* `dvc_dataset_hash`: MD5 hash of the feature store parquet file from `dvc.lock`.
* `dataset_version`: Dataset identifier (e.g. `v7`).
* `run_id`: MLflow tracking run UUID containing metrics, serialized artifacts, and feature importances.
* `loaded_at`: ISO 8601 UTC timestamp of model instantiation.

### 7.2 Gate Decision Logic ([`configs/gate.yaml`](configs/gate.yaml))
Newly trained candidates are registered under the `@challenger` alias. The automated promotion gate evaluates the challenger against the incumbent `@champion` on the identical holdout partition:

```
                       ┌─────────────────────────────┐
                       │   Candidate Model Trained   │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │    Register as @challenger  │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │ Holdout Evaluation vs. @champion  │
                    └─────────────────┬─────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 ▼                                         ▼
   [ MAE Gain >= 3.0%? ]                    [ Shortfall Surge <= 5.0%? ]
                 │                                         │
        YES ─────┴───── YES                       NO ──────┴────── NO
                 │                                         │
                 ▼                                         ▼
   ┌───────────────────────────┐             ┌───────────────────────────┐
   │ Promote to @champion      │             │ Reject Candidate          │
   │ Archive old as            │             │ Retain current @champion  │
   │ @previous_champion        │             └───────────────────────────┘
   └───────────────────────────┘
```

#### CLI Execution Commands
```bash
# Register candidate into registry
python -m src.registry.register

# Evaluate promotion gate with automated approval for CI
python -m src.registry.promote --auto-approve

# Instant rollback to a previous version if an operational issue is detected
python -m src.registry.rollback --to 1
```

---

## 8. Serving Infrastructure & Frontend Dashboard

### 8.1 Zero-Downtime Dynamic Model Hot-Reloading ([`service/app/model_loader.py`](service/app/model_loader.py))
To prevent dropped connections during model updates, the `DynamicModelLoader` implements thread-safe dynamic resolution:
1. Queries the MLflow Model Registry for the version tagged with `@champion`.
2. Compares the active version with the currently cached in-memory model.
3. If a new version is detected, loads the new model into a staging pointer, verifies an inference pass, and atomically swaps the reference.
4. If registry resolution encounters a cross-platform path disparity (e.g., Windows host path in SQLite vs. Linux container path), falls back to direct relative artifact resolution in `mlruns/`.

### 8.2 Production REST API Specification

| Endpoint | Method | Input Parameters / Body | Response Schema | Description |
| :--- | :---: | :--- | :--- | :--- |
| `/` | `GET` | None | `text/html` | Serves the Glassmorphism interactive web dashboard |
| `/health` | `GET` | None | `HealthResponse` | Health status (`healthy`), server uptime (seconds), active champion version |
| `/model-info` | `GET` | None | `ModelInfoResponse` | End-to-end lineage: Git commit, DVC hash, dataset version, MLflow run ID |
| `/stations` | `GET` | None | `list[StationItem]` | List of 30 stations with IDs, names, and hub status flags |
| `/forecast` | `GET` | `station_id` (str), `date` (YYYY-MM-DD) | `ForecastResponse` | 24-hour departure forecast, morning/evening peak hours, baseline, rebalancing advice |
| `/predict` | `POST` | `PredictRequest` (JSON features list) | `PredictResponse` | Batch inference endpoint returning raw departure predictions |
| `/docs` | `GET` | None | Swagger UI | Interactive OpenAPI documentation and API testing interface |

### 8.3 Operational Dashboard & Dispatch Alerts ([`app/static/index.html`](app/static/index.html))
The frontend dashboard features a modern dark-mode Glassmorphism UI built with Chart.js:
* **Real-Time Provenance Header:** Displays the serving model version, dataset tag, and Git commit SHA.
* **Station & Date Presets:** Instant selection across all 30 top stations and historical/live dates.
* **Hourly Departure Trajectory Chart:** Visualizes the predicted 24-hour curve with toggleable comparison against the 168-hour seasonal-naive baseline.
* **Rebalancing Recommendations:** Automatically calculates peak morning (06:00–11:00) and evening (16:00–21:00) spikes, generating actionable alerts:
  * *High Demand:* "Recommend staging at least 41 bicycles before morning peak commute (08:00)."
  * *Moderate Demand:* "Stage 12 bicycles prior to 08:00 morning peak."
  * *Normal Baseline:* "Maintain standard float of 5 bicycles."
* **MLOps Provenance Modal:** Clicking the lineage badge opens a modal showing holdout MAE, run ID, and exact loaded timestamp.

---

## 9. Containerization & Production DevOps Engineering

### 9.1 Container Optimization & Lightweight Serving Image
Production microservices should not bundle heavy training dependencies, compilers, or offline evaluation frameworks. BikeSense AI uses a dedicated manifest for serving:

* **Training Dependencies (`requirements.txt`):** Includes DVC, AWS CLI, scikit-learn, Evidently AI, Optuna, and plotting libraries (~1.7 GB unpruned container footprint).
* **Serving Dependencies (`requirements-serve.txt`):** Strips all offline tools, retaining only FastAPI, Uvicorn, Pandas, PyArrow, LightGBM, and `mlflow-skinny` + `sqlalchemy` (~387 MB final container image).

### 9.2 Robust SQLite Path Normalization
When MLflow logs runs on a Windows development machine, file URIs are stored in `mlflow.db` as `file:D:/Projects/bikesense-ai/mlruns/...`. When built into a Linux container (`python:3.11-slim`), these absolute host paths become invalid.

The production [`Dockerfile`](Dockerfile) includes an automated SQLite path normalizer:
```dockerfile
RUN python -c "\
import sqlite3; \
conn = sqlite3.connect('/app/mlflow.db'); \
cur = conn.cursor(); \
cur.execute(\"UPDATE experiments SET artifact_location = REPLACE(REPLACE(artifact_location, 'file:D:/Projects/bikesense-ai/mlruns', 'file:///app/mlruns'), 'file:///D:/Projects/bikesense-ai/mlruns', 'file:///app/mlruns')\"); \
cur.execute(\"UPDATE runs SET artifact_uri = REPLACE(REPLACE(artifact_uri, 'file:D:/Projects/bikesense-ai/mlruns', 'file:///app/mlruns'), 'file:///D:/Projects/bikesense-ai/mlruns', 'file:///app/mlruns')\"); \
conn.commit(); \
conn.close()"
```

### 9.3 Synthetic MLflow Schema Initializer ([`service/init_stub_mlflow.py`](service/init_stub_mlflow.py))
Because trained model binaries and database files are gitignored, fresh CI/CD runner checkouts lack `mlflow.db`. Rather than failing container builds, [`service/init_stub_mlflow.py`](service/init_stub_mlflow.py) initializes a clean MLflow SQLite schema using native MLflow client calls, ensuring that Alembic migrations match perfectly and preventing `duplicate column name` database collisions.

---

## 10. Continuous Integration & Continuous Deployment (CI/CD)

The project includes three GitHub Actions workflows in [`.github/workflows/`](.github/workflows/):

### 10.1 CI Pipeline (`ci.yml`)
* **Job 1 (Linting):** Runs `ruff check src/ app/ service/ tests/`.
* **Job 2 (Automated Tests):** Runs all 38 pytest test cases with XML code coverage generation.
* **Job 3 (Docker Build & Health Verification):** Sets up Docker Buildx, creates synthetic MLflow stubs, builds the production container, runs it locally, and verifies that `curl /health` and `curl /` return HTTP 200.

### 10.2 CD Pipeline (`cd.yml`)
* **AWS Authentication:** Connects via `aws-actions/configure-aws-credentials@v4` using `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` secrets, defaulting to `us-east-1` if `AWS_REGION` is omitted.
* **Amazon ECR:** Logs into ECR, verifies repository existence (or creates it), builds the multi-platform container image, and pushes both `:${{ github.sha }}` and `:latest` tags.
* **Amazon EC2 Deployment:** Securely connects over SSH, pulls the updated image from Amazon ECR, terminates the old container, launches the new container on port 8000, and verifies health checks.

### 10.3 Automated Retraining Pipeline (`retrain.yml`)
* **Trigger:** Scheduled cron execution every Sunday at midnight UTC (`0 0 * * 0`) or via manual `workflow_dispatch`.
* **Execution:** Pulls latest dataset version from Amazon S3 DVC remote, runs `dvc repro`, registers candidate models, executes the promotion gate with `--auto-approve`, and pushes updated artifacts back to S3.

---

## 11. Operational Monitoring & Statistical Drift Detection

### 11.1 Delayed Ground Truth Architecture ([`src/monitoring/join_actuals.py`](src/monitoring/join_actuals.py))
Every live inference request is logged to `data/predictions/prediction_log.csv` with a unique timestamp, station ID, and predicted departure values. Twenty-four hours later, the actual observed trip departures are extracted from Citi Bike feeds and joined against the predictions to calculate real-world operational error without manual labeling.

### 11.2 Evidently AI Statistical Drift Testing ([`src/monitoring/drift_report.py`](src/monitoring/drift_report.py))
* **Continuous Meteorological Drift:** Evaluates distribution shifts in temperature, precipitation, and wind speed using two-sample **Kolmogorov-Smirnov (K-S)** tests ($\alpha = 0.05$).
* **Temporal Demand Drift:** Uses **Wasserstein Distance** to detect structural changes in lag features (`lag_24h`, `lag_168h`), alerting dispatchers to macro ridership shifts.

### 11.3 Multi-Signal Alert Definitions ([`src/monitoring/alerts.py`](src/monitoring/alerts.py))
* **`ALERT-101 (Covariate Drift):`** Triggered when $> 30\%$ of input features exhibit statistically significant drift over a 7-day rolling window.
* **`ALERT-102 (Accuracy Degradation):`** Triggered when rolling 7-day holdout MAE exceeds $1.25\times$ reference training MAE.
* **`ALERT-103 (Stockout Risk Alert):`** Triggered when under-forecast shortfall increases by $> 15\%$ over a 48-hour period, indicating severe station starvation risk.

---

## 12. Empirical Benchmark Evaluation

Models evaluated on the 36-month (`v7`) holdout replay window (July – August 2026, 44,640 station-hour observations):

| Model Name | Version Tag | Architecture Details | Holdout MAE | Holdout RMSE | Shortfall (Bikes) | Gate Decision |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Seasonal-Naive Baseline** | — | Persistence Heuristic ($t - 168\text{h}$) | 1.8410 | 2.6512 | 19,840.00 | Reference Benchmark |
| **Ridge Regression** | — | L2 Regularized ($\alpha = 1.0$) | 1.6240 | 2.3110 | 17,210.50 | Baseline ML Model |
| **LightGBM Shallow** | `v1` | `max_depth=5, num_leaves=31` | 1.3864 | 2.0411 | 13,934.50 | Displaced Champion |
| **LightGBM Tuned (Trial 3)**| `v2` | `max_depth=7, num_leaves=63, lr=0.06` | 1.3992 | 2.0513 | 13,963.28 | Rejected (Insufficient gain) |
| **LightGBM Tuned (Trial 2)**| **`v3`** | **`max_depth=6, num_leaves=45, lr=0.04`** | **1.3153** | **2.0037** | **14,179.34** | 🏆 **PROMOTED TO @champion (+5.13% gain)** |

### Performance Highlights
* **+28.5% MAE Improvement** over the Seasonal-Naive baseline heuristic.
* **+5.13% MAE Improvement** over the initial champion model (`v1`), comfortably exceeding the 3.0% promotion threshold.
* **Shortfall Contained:** Under-forecast shortfall remained well within the 5.0% maximum allowable surge guardrail (+1.75%), preventing starvation events across all 30 stations.

---

## 13. Testing, Verification & Quality Assurance Matrix

The repository maintains an automated test suite of **38 unit and integration tests** located in [`tests/`](tests/):

```bash
# Execute test suite with summary report
pytest tests/ -v --tb=short

# Execute with coverage report
pytest tests/ --cov=src --cov=app --cov-report=term-missing
```

| Test Directory | Focus Area | Test Count | Key Invariants Verified |
| :--- | :--- | :---: | :--- |
| `tests/data/` | Ingest, Weather, Holidays, QC | 7 | Non-negative departures, full Cartesian grid completeness, date boundary adherence, weather caching |
| `tests/features/` | Feature Engineering | 5 | Temporal feature calculations, strict historical lag ordering, zero temporal data leakage |
| `tests/models/` | Training & Metrics | 9 | MAE/RMSE/Shortfall formula accuracy, time-ordered train/test splits, baseline heuristic reproducibility |
| `tests/registry/` | MLflow & Promotion Gate | 6 | Gate approval/rejection logic, shortfall guardrail enforcement, zero-downtime rollback integrity |
| `tests/service/` | FastAPI & Model Loader | 5 | `/health`, `/forecast`, `/model-info`, `/predict` schemas, dynamic model loading, prediction logging |
| `tests/monitoring/`| Drift & Alert Engine | 6 | Evidently report generation, K-S drift detection, alert trigger thresholds, delayed actuals join |

---

## 14. Conclusion & Operational Impact

BikeSense AI establishes a modern blueprint for production spatio-temporal MLOps. By coupling reproducible data versioning (DVC) with robust model governance (MLflow), zero-downtime dynamic serving (FastAPI), automated container delivery (Docker + AWS ECR/EC2), and statistical drift monitoring (Evidently AI), the platform ensures that Citi Bike dispatchers have continuous, reliable foresight into hourly demand, minimizing station starvation, reducing wasted truck mileage, and improving commuter satisfaction across the network.
