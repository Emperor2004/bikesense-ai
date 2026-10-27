# BikeSense AI: Drift-Aware Bike-Share Demand Forecasting

A versioned and monitored MLOps pipeline that forecasts hourly station-level demand for Citi Bike (New York City) and keeps the forecasting model trustworthy over time.

**Course:** MLOps (final-year project) · **Infrastructure cost:** USD 0 (open data, open-source tools) · **Duration:** 8 weeks

---

## 1. What this project is

Bike-share operators must move bikes between stations before demand arrives. A forecasting model trained once degrades as seasons, weather and rider behaviour change, and the degradation stays invisible until predictions are compared with the ground truth that arrives later.

This project builds a **24-hour, station-level departure forecasting service** and, as its core contribution, the controls around it:

| Pillar | Question it answers | Tool |
|---|---|---|
| Data versioning | Can every dataset be reproduced exactly? | DVC + Git |
| Model versioning | Which data and code produced the model now in service? Can we roll back? | MLflow tracking + registry |
| Monitoring | Is the model still accurate this month? | Evidently (Prometheus + Grafana optional) |
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
Train & evaluate (baseline, Ridge, LightGBM)
        │
        ▼
Model versioning (MLflow registry, champion / challenger aliases)
        │
        ▼
Serving (FastAPI in Docker, loads the champion alias)
        │
        ▼
Prediction log ──joined with actual demand 24 h later──► Monitoring (Evidently)
                                                              │
                                                              ▼
                                                  Alert & retrain workflow
```

## 5. Tech stack

| Layer | Tool |
|---|---|
| Data versioning | DVC (local remote) + Git |
| Tracking and registry | MLflow (local server, SQLite backend) |
| Modeling | scikit-learn, LightGBM |
| Serving | FastAPI, Docker Compose |
| Monitoring | Evidently; Prometheus + Grafana (stretch) |
| Automation | CI checks and a retrain workflow |

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
├── README.md
├── requirements.txt
├── Makefile
├── dvc.yaml                  # pipeline: ingest -> features -> train
├── dvc.lock                  # pinned content hashes per dataset version
├── params.yaml
├── configs/                  # features, model, promotion gate, monitoring
├── data/                     # raw, external (weather), processed  (DVC-tracked)
├── notebooks/                # EDA and station selection
├── src/
│   ├── data/                 # ingest, validate, weather, holidays
│   ├── features/             # feature builder
│   ├── models/               # baseline, train, evaluate, metrics
│   ├── registry/             # register, promote, rollback
│   ├── replay/               # replay driver (simulated production time)
│   └── monitoring/           # actuals join, rolling error, drift report, alerts
├── service/                  # FastAPI app and Dockerfile
├── monitoring/               # Prometheus and Grafana config (optional)
├── docker-compose.yml
├── tests/                    # data, features, models, registry, service, monitoring
├── scripts/                  # registry audit, fresh-clone test, demo runner
├── reports/                  # evaluation and monitoring outputs
└── docs/                     # architecture, lineage report, model card, slides
```

## 9. Getting started

**Prerequisites:** Python 3.10+, Docker and Docker Compose, Git, DVC, about 10 GB free disk.

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Reproduce data and model pipeline for the current dataset version
dvc repro

# 3. Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

# 4. Start the forecasting service and monitoring stack
docker compose up --build
```

Convenience targets:

```bash
make data         # run ingest + features
make train        # train baseline, Ridge, LightGBM and log to MLflow
make register     # register best run as challenger
make serve        # start the FastAPI service
make replay       # replay the next window as simulated production
make monitor      # generate Evidently reports
make test         # run the unit and schema tests
make demo         # run the end-to-end demonstration
```

## 10. Using the service

```bash
# Health check
curl http://localhost:8000/health

# 24-hour forecast for one station
curl "http://localhost:8000/forecast?station_id=<STATION_ID>&date=<YYYY-MM-DD>"

# Which model is serving, and where did it come from?
curl http://localhost:8000/model-info
```

`/model-info` returns the champion version, Git commit and dataset hash. Every prediction is appended to the prediction log and later joined with actual demand.

## 11. Model versioning and promotion

- Every run logs parameters, metrics, feature list, Git commit, dataset version (DVC hash) and the model artifact.
- The best run is registered as **challenger**. The service always loads the **champion** alias, so promotion and rollback are alias changes, not code redeployments.
- **Promotion gate:** a challenger replaces the champion only if its MAE on the latest held-out month is lower by a configured margin (initially 3%, set in `configs/gate.yaml`). A human approves the alias change.

```bash
python -m src.registry.promote --challenger <VERSION>
python -m src.registry.rollback --to <VERSION>     # one-step rollback, no redeploy
```

## 12. Monitoring

During replay the service issues a forecast each day. Actual demand is joined to the prediction log 24 hours later. The reference dataset is the training data of the current champion.

| Layer | Signal | Method | Action |
|---|---|---|---|
| Input drift | Weather, calendar, lag features vs reference | Evidently data drift report | Flag when share of drifted features exceeds threshold |
| Prediction drift | Distribution of predicted demand | Evidently | Informational |
| Model performance | Rolling 7-day MAE vs reference MAE | Evidently or custom metric | Alert above 1.25x reference (configurable) |
| Service health | Latency, request count, errors | Prometheus + Grafana (stretch) | Informational |

Reports are written to `reports/monitoring/` as HTML.

## 13. Demonstration flow

1. Champion trained on v1 serves forecasts; show its lineage (Git commit and data hash).
2. Replay the next window: monitoring shows input drift and rising rolling MAE.
3. Release dataset v2 and run the retrain workflow; compare challenger and champion in MLflow.
4. Apply the promotion gate and switch the champion alias; the service now serves the new model.
5. Roll back by repointing the alias and show recovery without a code change.

## 14. Success criteria

| Criterion | Target |
|---|---|
| Forecast quality | Champion beats the seasonal-naive baseline on MAE on each replay window |
| Business proxy | Under-forecast shortfall reported next to MAE for every window and model version |
| Lineage | 100% of registered model versions carry Git commit and dataset version |
| Reproducibility | A clean copy running `dvc repro` reproduces registered metrics within a small tolerance |
| Detection | An alert fires within one monitoring window (7 days) of a visible degradation |
| Rollback | Reverting the champion alias restores service in under 5 minutes, no redeploy |

## 15. Known limitations

- **Censored demand:** trip data understates demand where stations were empty.
- **Weather optimism:** historical weather is reanalysis (ERA5-based), which a live system would replace with forecasts, so accuracy is likely optimistic.
- **Drift depends on the chosen months:** observed drift is reported honestly; if real drift is weak, a clearly labelled injected shift is used as a secondary test only.
- **Local deployment only:** no Kubernetes, cloud, feature store, streaming feed, rebalancing optimisation or deep learning.

## 16. References

- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS 28.
- Citi Bike System Data and NYCBS Data Use Policy: https://citibikenyc.com/system-data
- Open-Meteo Historical Weather API (CC BY 4.0): https://open-meteo.com
- Ataç, S., Obrenović, N., Bierlaire, M. (2020). *Vehicle sharing systems: Does demand forecasting yield a better service?* STRC.
- *Predictive and Prescriptive Performance of Bike-Sharing Demand Forecasts for Inventory Management.* arXiv:2108.00858.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *QJRMS*, 146(730), 1999-2049.
- Official documentation: DVC, MLflow, Evidently AI.

## 17. Team

| Role | Focus |
|---|---|
| Member A: Data and automation | Ingestion, features, DVC pipeline, dataset versions, replay, CI |
| Member B: Modeling and registry | Models, evaluation, MLflow registry, promotion gate, rollback |
| Member C: Serving and monitoring | FastAPI service, Docker, prediction log, Evidently, alerts |
