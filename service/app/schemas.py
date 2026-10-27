"""Pydantic data schemas and API contracts for BikeSense AI serving service."""


from pydantic import BaseModel, Field


class HourlyForecast(BaseModel):
    """Hourly forecast prediction item."""

    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    predicted_departures: float = Field(..., ge=0.0, description="Predicted departure demand")


class StationItem(BaseModel):
    """Citi Bike Station representation."""

    id: str = Field(..., description="Unique station alphanumeric identifier")
    name: str = Field(..., description="Human-readable station street/hub name")
    is_hub: bool = Field(False, description="Flag indicating if station is a primary transit hub")


class ForecastResponse(BaseModel):
    """Response schema for a 24-hour departure forecast request."""

    station_id: str
    station_name: str | None = None
    date: str
    model_version: str
    git_commit: str
    dvc_dataset_hash: str
    issued_at: str
    total_predicted_departures: float
    forecasts: list[HourlyForecast]
    baseline_forecasts: list[HourlyForecast] | None = None
    peak_morning_hour: int | None = None
    peak_evening_hour: int | None = None
    rebalancing_recommendation: str | None = None


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
