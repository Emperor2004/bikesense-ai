# BikeSense AI — Running Backend, Frontend & MLflow Servers

This guide details how to start and interact with the **Full MLOps Stack (Option 2)**: the **FastAPI Backend**, the **Interactive Frontend Web Dashboard**, and the **MLflow Model Registry & Tracking Server**.

---

## 1. Quick Access Overview

| Component | URL | Purpose |
|---|---|---|
| **Frontend Web Dashboard** | [http://localhost:8000](http://localhost:8000) | Interactive demand curves, station selector, date picker, model metadata |
| **Backend REST API (Swagger)** | [http://localhost:8000/docs](http://localhost:8000/docs) | OpenAPI interactive endpoint exploration & testing |
| **Backend Alternative Docs (ReDoc)**| [http://localhost:8000/redoc](http://localhost:8000/redoc) | Alternative structured API documentation |
| **MLflow Tracking & Registry UI** | [http://localhost:5000](http://localhost:5000) | Experiment runs, hyperparameter tuning comparison, `@champion` alias |

---

## 2. Prerequisites

Activate your project Python virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

---

## 3. Starting the Servers

### Terminal 1: Start the MLflow Tracking Server

Run the MLflow tracking UI connected to the local SQLite database and model artifact repository:

```powershell
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

> **What it does:**
> - Serves the web UI on port `5000`.
> - Allows visual comparison of all DVC experiment runs (`funny-vina`, `exp-seed-101`, `exp-seed-202`).
> - Displays registered model versions for `bikesense-demand-forecaster` and model tags (Git commit, DVC hash).

---

### Terminal 2: Start the FastAPI Backend & Frontend Dashboard

Start the Uvicorn ASGI server hosting both the REST API and the frontend dashboard:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

> **What it does:**
> - Automatically warm-loads the active `@champion` model from MLflow into memory.
> - Serves the frontend web dashboard on `/` (`http://localhost:8000`).
> - Serves the forecasting and monitoring REST API endpoints on port `8000`.

---

## 4. API Endpoints & Usage

### 1. Healthcheck
Verify service uptime and the currently serving champion model version:
```bash
curl http://localhost:8000/health
```
**Example Response:**
```json
{
  "status": "healthy",
  "model_name": "bikesense-demand-forecaster",
  "active_champion_version": "1",
  "uptime_seconds": 45.2
}
```

### 2. Model Lineage & Metadata
Check the Git commit and DVC dataset hash tied to the serving model:
```bash
curl http://localhost:8000/model-info
```

### 3. Generate 24-Hour Station Departure Forecast
Get the 24-hour demand prediction for a specific station and date:
```bash
curl "http://localhost:8000/forecast?station_id=JC014&date=2024-03-15"
```

### 4. Direct JSON Batch Inference (`/predict`)
Submit raw feature values for immediate model scoring:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {
        "hour": 8,
        "day_of_week": 1,
        "month": 7,
        "is_holiday": 0,
        "temperature_2m": 22.5,
        "precipitation": 0.0,
        "wind_speed_10m": 12.0,
        "lag_24h": 14.0,
        "lag_168h": 15.0,
        "rolling_7d_same_hour_mean": 14.2
      }
    ]
  }'
```

---

## 5. Stopping the Servers

To stop either server when running in the foreground terminal:
- Press `Ctrl + C` in the respective terminal window.

To check and terminate background processes using PowerShell:
```powershell
# Stop process on port 8000 (FastAPI)
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

# Stop process on port 5000 (MLflow)
Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```
