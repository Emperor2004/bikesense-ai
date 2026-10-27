"""Pydantic data schemas and API contracts for BikeSense AI serving service."""


from pydantic import BaseModel, Field


class HourlyForecast(BaseModel):
    """Hourly forecast prediction item."""

    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    predicted_departures: float = Field(..., ge=0.0, description="Predicted departure demand")


class ForecastResponse(BaseModel):
    """Response schema for a 24-hour departure forecast request."""

    station_id: str
    date: str
    model_version: str
    git_commit: str
    dvc_dataset_hash: str
    issued_at: str
    total_predicted_departures: float
    forecasts: list[HourlyForecast]


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    model_name: str
    active_champion_version: str | None = None
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Lineage and metadata response schema for the currently loaded champion model."""

    model_name: str
    champion_version: str
    run_id: str
    git_commit: str
    dvc_dataset_hash: str
    dataset_version: str
    loaded_at: str
