# BikeSense AI — MVP Demonstration & Presentation Guide

This guide provides a structured, 10-minute end-to-end demonstration flow to showcase the **BikeSense AI** production MLOps system. It is designed for live walkthroughs, academic evaluations, stakeholder demos, and technical interviews.

---

## 🎬 1-Minute Elevator Pitch

> *"BikeSense AI is an enterprise-grade, drift-aware MLOps platform for urban mobility. Bike-share operators lose revenue when stations are empty and anger riders when stations are full. While standard ML models degrade rapidly due to weather shifts and seasonality, BikeSense AI builds a closed-loop lifecycle around the model: versioning 38 months of Citi Bike trip data with DVC, automating hyperparameter sweeps with DVC Experiments, enforcing a quantitative MLflow promotion gate, and serving 24-hour departure forecasts via a FastAPI dashboard with zero-downtime hot-reloading."*

---

## 📋 Pre-Demo Checklist & Environment Setup

Verify both background servers are running before beginning your presentation:

| Service | URL | Purpose |
| :--- | :--- | :--- |
| **FastAPI Dashboard & REST API** | [http://127.0.0.1:8000/](http://127.0.0.1:8000/) | Live interactive operational forecasting UI |
| **FastAPI OpenAPI Swagger Docs** | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | Interactive API exploration console |
| **MLflow Tracking & Registry UI** | [http://127.0.0.1:5000/](http://127.0.0.1:5000/) | Model runs, metrics, parameters, and registry |

If not running, launch them with:
```powershell
# Terminal 1: MLflow Server
.\.venv\Scripts\python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000

# Terminal 2: FastAPI Microservice
powershell -Command "$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; .\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload"
```

---

## 🚀 Live Demonstration Flow (Step-by-Step)

### Phase 1: Interactive Operational Dashboard (3 Minutes)

1. **Open the Dashboard:**
   Navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your web browser.
   
2. **Highlight Top KPI Cards:**
   * Point out the **Active Serving Model**: Shows the currently loaded model architecture.
   * Point out the **Alias Badge**: Highlights `@champion (Version 3)`.
   * Point out the **Total Daily Demand** and **Morning/Evening Commute Peaks**.

3. **Demonstrate Full MLOps Provenance (The Lineage Modal):**
   * Click the top-right green badge: **"MLflow Champion Active"**.
   * A glassmorphic modal opens displaying the complete audit trail:
     * **Model Name:** `bikesense-demand-forecaster`
     * **Registry Version:** `Version 3`
     * **Dataset Span:** `v7 (38 Months: 2023-07 to 2026-08)`
     * **Git Commit Hash:** Exact commit SHA generating this artifact
     * **DVC Dataset Hash:** Content hash of the feature store parquet
     * **Holdout MAE:** `1.3153` (`+5.13%` improvement over v1)
   * *Key Takeaway:* Every prediction can be traced back to the exact code commit, data slice, and hyperparameter configuration that produced it.

4. **Demonstrate One-Click Scenario Presets:**
   Click the preset buttons under the control panel to show how the model responds to real-world operational scenarios:
   * 🌞 **Summer Commute Peak (Grove St):** High morning and evening commuter spikes. Point out the dynamic **Operational Rebalancing Directive** recommending proactive bike staging (e.g. *"Stage 47 bicycles before peak intervals"*).
   * 🚆 **Ferry & Transit Hub (Hoboken Terminal):** Demonstrates multi-modal transit connection surges.
   * ❄️ **Winter Cold Shift (Newport PATH):** Shows how the model adapts to freezing temperatures and dampens departure demand without manual intervention.
   * 🎆 **Holiday Recreation (Hamilton Park):** Demonstrates midday leisure patterns on Independence Day rather than dual commuter peaks.

5. **Compare ML Forecast vs 168h Baseline:**
   * Toggle the **"Compare 168h Baseline Heuristic"** checkbox.
   * *Talking Point:* Explain that the amber dashed line is what a naive operator would do (copy demand from 1 week ago). Point out how LightGBM smooths out noisy outliers and adjusts for weather while retaining diurnal sharpness.

---

### Phase 2: DVC Hyperparameter Experiments & Auto-Apply (2 Minutes)

1. **Show DVC Experiment Tracking in the Terminal:**
   ```bash
   dvc exp list
   ```
   *Output shows the 4 experiment branches:*
   * `exp_seed_42_a1`
   * `exp_seed_101_a3`
   * `exp_seed_777_a05`
   * `exp_seed_999_a5`

2. **Explain the Automated Multi-Experiment Runner:**
   Show [scripts/run_experiments_and_apply.py](scripts/run_experiments_and_apply.py).
   * It executed 4 distinct hyperparameter configurations using `dvc exp run -S ...`.
   * It extracted the holdout MAE across all candidates:
     * `exp_seed_42_a1`: `1.3211`
     * `exp_seed_101_a3`: `1.3183`
     * **`exp_seed_777_a05`**: **`1.3153` (WINNER)**
     * `exp_seed_999_a5`: `1.3210`
   * It automatically applied the winning experiment to the workspace (`dvc exp apply`).
   * It registered the winning candidate in MLflow and triggered the promotion gate.

---

### Phase 3: MLflow Model Registry & Promotion Gate (2 Minutes)

1. **Open the MLflow Tracking Server:**
   Navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).
   * Click on **Models** in the top navigation bar.
   * Click on **`bikesense-demand-forecaster`**.

2. **Explain the 3 Registered Versions:**
   * **Version 1:** Initial model trained on dataset v1. Displaced champion, now tagged as `@previous_champion`.
   * **Version 2:** Challenger model from retrain. **Rejected** by the promotion gate because its improvement was under the required 3.0% margin (-0.92%).
   * **Version 3:** Promoted **`@champion`**! Succeeded with `+5.13%` MAE improvement over the baseline on holdout month 2026-06.

3. **Demonstrate Zero-Downtime Hot-Reloading:**
   * Show that while the server was running, Version 3 was promoted.
   * Call the endpoint:
     ```bash
     curl http://127.0.0.1:8000/model-info
     ```
   * Notice `"champion_version": "3"` without any server restart or dropped requests!

4. **Demonstrate Instant Rollback (Optional Demo Hook):**
   ```bash
   # Instantly rollback to version 1
   python -m src.registry.rollback --to 1
   
   # Verify the service immediately reloaded version 1
   curl http://127.0.0.1:8000/model-info
   
   # Restore version 3
   python -m src.registry.rollback --to 3
   ```

---

### Phase 4: CI/CD & Production Infrastructure (2 Minutes)

Open `.github/workflows/` in your code editor:

1. **`ci.yml` (Continuous Integration):**
   * Runs on every push/PR to `main`.
   * Executes Ruff linting, pytest (38 unit/integration tests), and dry-run DVC reproduction.
2. **`cd.yml` (Continuous Deployment):**
   * Builds the Docker image upon merge to `main`.
   * Pushes the image to **Amazon ECR**.
   * Deploys directly to an **Amazon EC2** instance via SSH and restarts the container via Docker Compose.
3. **`retrain.yml` (Automated Continuous Training):**
   * Runs weekly every Sunday at 02:00 UTC.
   * Pulls latest data from S3, reruns DVC, and evaluates candidates through the promotion gate.

---

### Phase 5: Monitoring & Drift Detection (1 Minute)

1. **Delayed Ground-Truth Architecture:**
   * Show `data/predictions/prediction_log.csv`.
   * Predictions issued at $T=00:00$ are logged immediately.
   * Actual Citi Bike departures are joined 24 hours later using [src/monitoring/join_actuals.py](src/monitoring/join_actuals.py).

2. **Evidently AI & Multi-Signal Alerting:**
   * **Input Drift (`ALERT-101`):** Flags when $> 30\%$ of weather/calendar features drift using Kolmogorov-Smirnov tests.
   * **Performance Drift (`ALERT-102`):** Triggers when rolling 7-day MAE exceeds $1.25\times$ reference MAE.
   * **Operational Risk (`ALERT-103`):** Triggers when under-forecast shortfall surges, preventing empty station stockouts.

---

## 🎯 Evaluator Q&A Cheat Sheet

| Question | Strong Answer |
| :--- | :--- |
| **How did you prevent temporal data leakage?** | *"All train/val splits are strictly time-ordered without shuffling. In `features.parquet`, lag features (`lag_24h`, `lag_168h`) and rolling means only use data available prior to the forecast issue time (00:00 UTC). Weather data is aligned hour-by-hour."* |
| **Why LightGBM over Deep Learning (LSTM/Transformers)?** | *"Tabular time series with rich calendar and exogenous meteorological features consistently perform better and train orders of magnitude faster with GBDTs than neural networks. LightGBM trains in under 15 seconds, making continuous retraining and multi-trial DVC sweeps practical."* |
| **What happens if a new station is added (cold-start)?** | *"The top 30 stations were locked from dataset v1 to ensure a consistent spatial basis. For brand-new stations, the service can fall back to system-wide diurnal averages or nearest-station feature imputation."* |
| **Why is Under-Forecast Shortfall reported alongside MAE?** | *"MAE is symmetric—it treats under-predicting and over-predicting equally. But in bike-share operations, under-predicting is far worse because it causes empty stations and lost rides. Shortfall directly quantifies unserved demand units."* |
| **How do you handle memory constraints in production?** | *"We set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` across all scripts and container runtimes to prevent BLAS heap exhaustion on multi-core servers."* |
