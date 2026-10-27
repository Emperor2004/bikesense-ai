"""Main FastAPI entrypoint for BikeSense AI serving microservice.

Provides REST API endpoints for:
- /health: Service health and champion model status
- /model-info: Model metadata, dataset version, and Git lineage
- /forecast: 24-hour station-level departure demand forecast
- /predict: Direct batch/single JSON inference
"""

import os

# Limit OpenBLAS concurrency threads to prevent memory allocation exhaustion on Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from service.app.model_loader import DynamicModelLoader
from service.app.prediction_log import PredictionLogger
from service.app.schemas import (
    ForecastResponse,
    HealthResponse,
    HourlyForecast,
    ModelInfoResponse,
    StationItem,
)

START_TIME = time.time()
loader = DynamicModelLoader()
logger = PredictionLogger()

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "month",
    "is_holiday",
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "lag_24h",
    "lag_168h",
    "rolling_7d_same_hour_mean",
]


class PredictRequest(BaseModel):
    """Direct inference request schema."""

    features: list[dict[str, Any]] = Field(
        ...,
        description="List of feature records matching the model inputs.",
        examples=[
            [
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
                    "rolling_7d_same_hour_mean": 14.2,
                }
            ]
        ],
    )


class PredictResponse(BaseModel):
    """Direct inference response schema."""

    model_version: str
    predictions: list[float]
    count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up and pre-load the active champion model on application startup."""
    try:
        loader.load_champion_model()
    except Exception as e:
        print(f"[Startup Warning] Could not preload champion model: {e}")
    yield


app = FastAPI(
    title="BikeSense AI Demand Forecasting Service",
    description=(
        "Production-ready FastAPI service for Citi Bike hourly departure demand forecasting. "
        "Integrates with MLflow Model Registry and supports EC2 deployment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Frontend"])
def root():
    """Serves the interactive demand forecasting web dashboard."""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "service": "BikeSense AI Demand Forecasting Service",
        "docs_url": "/docs",
        "health_url": "/health",
        "forecast_url": "/forecast",
    }


@app.get("/api-info", tags=["General"])
def api_info():
    """Returns general service information and endpoint links."""
    return {
        "service": "BikeSense AI Demand Forecasting Service",
        "docs_url": "/docs",
        "health_url": "/health",
        "forecast_url": "/forecast",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Service health check returning uptime and the loaded champion model version."""
    champ_ver = loader.get_champion_version()
    return HealthResponse(
        status="healthy",
        model_name=loader.model_name,
        active_champion_version=str(champ_ver) if champ_ver is not None else None,
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
def model_info():
    """Returns detailed lineage and metadata for the serving champion model."""
    try:
        _, metadata = loader.load_champion_model()
        return ModelInfoResponse(**metadata)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e!s}")


def get_feature_matrix_for_date(station_id: str, date_str: str) -> pd.DataFrame:
    """Retrieve or construct the 24-hour feature matrix for station and date."""
    features_path = Path("data/processed/features.parquet")
    if not features_path.exists():
        features_path = Path("data/processed/features_v1.parquet")

    if features_path.exists():
        df = pd.read_parquet(features_path)
        df["dt_str"] = pd.to_datetime(df["datetime_hour"]).dt.strftime("%Y-%m-%d")
        matched = df[(df["station_id"] == station_id) & (df["dt_str"] == date_str)].copy()
        if len(matched) == 24:
            matched = matched.sort_values("hour").reset_index(drop=True)
            return matched

    # Fallback diurnal construction when date is not in precomputed features
    dt = pd.to_datetime(date_str)
    hours = list(range(24))
    timestamps = [pd.to_datetime(f"{date_str} {h:02d}:00:00") for h in hours]

    data = {
        "station_id": [station_id] * 24,
        "datetime_hour": timestamps,
        "hour": hours,
        "day_of_week": [dt.day_of_week] * 24,
        "month": [dt.month] * 24,
        "is_holiday": [0] * 24,
        "temperature_2m": [18.0 + 4.0 * np.sin((h - 6) / 12 * np.pi) for h in hours],
        "precipitation": [0.0] * 24,
        "wind_speed_10m": [12.0] * 24,
        "lag_24h": [4.0] * 24,
        "lag_168h": [4.5] * 24,
        "rolling_7d_same_hour_mean": [5.0] * 24,
    }
    return pd.DataFrame(data)


@app.get("/stations", response_model=list[StationItem], tags=["Stations"])
def get_stations():
    """Retrieve list of all 30 top Citi Bike stations with human-readable names."""
    names_path = Path("data/processed/station_names.json")
    if names_path.exists():
        with open(names_path, "r", encoding="utf-8") as f:
            station_dict = json.load(f)
    else:
        station_dict = {"JC013": "Marin Light Rail", "JC014": "Grove St PATH"}

    hubs = {"JC115", "HB102", "HB101", "JC066", "JC116", "JC104"}
    items = [
        StationItem(id=sid, name=sname, is_hub=(sid in hubs))
        for sid, sname in station_dict.items()
    ]
    return sorted(items, key=lambda x: (not x.is_hub, x.name))


@app.get("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def forecast(
    station_id: str = Query(..., description="Canonical Citi Bike station identifier"),
    date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Forecast date (YYYY-MM-DD)"),
):
    """Generate and log a 24-hour departure forecast for a given station and date."""
    try:
        model, metadata = loader.load_champion_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e!s}")

    feature_df = get_feature_matrix_for_date(station_id, date)
    X = feature_df[FEATURE_COLS]

    raw_preds = model.predict(X)
    preds = np.maximum(0.0, raw_preds)

    issued_at = datetime.now(UTC).isoformat()
    hourly_forecasts = []
    log_items = []

    for h, p in enumerate(preds):
        p_rounded = round(float(p), 2)
        hourly_forecasts.append(HourlyForecast(hour=h, predicted_departures=p_rounded))
        dt_hour = f"{date} {h:02d}:00:00"
        log_items.append({"datetime_hour": dt_hour, "predicted": p_rounded})

    # Baseline forecasts: lag_168h represents seasonal-naive heuristic (same hour 1 week ago)
    baseline_vals = feature_df.get("lag_168h", pd.Series(preds)).to_numpy()
    baseline_forecasts = [
        HourlyForecast(hour=h, predicted_departures=round(float(max(0.0, b)), 2))
        for h, b in enumerate(baseline_vals)
    ]

    # Compute morning peak (hours 6 to 11) and evening peak (hours 16 to 21)
    morning_hours = list(range(6, 12))
    evening_hours = list(range(16, 22))
    peak_morning = max(morning_hours, key=lambda h: preds[h])
    peak_evening = max(evening_hours, key=lambda h: preds[h])

    # Formulate practical rebalancing recommendation
    peak_morning_val = preds[peak_morning]
    peak_evening_val = preds[peak_evening]
    if peak_morning_val >= 15.0 or peak_evening_val >= 15.0:
        rec = f"High demand alert: Recommend staging at least {int(np.ceil(max(peak_morning_val, peak_evening_val) * 1.2))} bicycles before peak commute ({peak_morning:02d}:00 / {peak_evening:02d}:00)."
    elif peak_morning_val >= 6.0:
        rec = f"Moderate demand: Stage {int(np.ceil(peak_morning_val))} bicycles prior to {peak_morning:02d}:00 morning peak."
    else:
        rec = "Normal operating baseline: Maintain standard float of 5 bicycles."

    # Look up station name
    station_names_file = Path("data/processed/station_names.json")
    station_name = station_id
    if station_names_file.exists():
        with open(station_names_file, "r", encoding="utf-8") as f:
            station_name = json.load(f).get(station_id, station_id)

    logger.log_forecast(
        station_id=station_id,
        issued_at=issued_at,
        forecasts=log_items,
        model_version=metadata["champion_version"],
        git_commit=metadata["git_commit"],
    )

    total_predicted = round(float(np.sum(preds)), 2)

    return ForecastResponse(
        station_id=station_id,
        station_name=station_name,
        date=date,
        model_version=metadata["champion_version"],
        git_commit=metadata["git_commit"],
        dvc_dataset_hash=metadata["dvc_dataset_hash"],
        issued_at=issued_at,
        total_predicted_departures=total_predicted,
        forecasts=hourly_forecasts,
        baseline_forecasts=baseline_forecasts,
        peak_morning_hour=peak_morning,
        peak_evening_hour=peak_evening,
        rebalancing_recommendation=rec,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict(payload: PredictRequest):
    """Direct batch prediction endpoint taking raw feature records."""
    try:
        model, metadata = loader.load_champion_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e!s}")

    df = pd.DataFrame(payload.features)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required feature columns: {missing}")

    X = df[FEATURE_COLS]
    raw_preds = model.predict(X)
    preds = [round(float(p), 3) for p in np.maximum(0.0, raw_preds)]

    return PredictResponse(
        model_version=metadata["champion_version"],
        predictions=preds,
        count=len(preds),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
