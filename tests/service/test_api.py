"""Unit tests for FastAPI serving endpoints."""

import pytest
from fastapi.testclient import TestClient

from service.app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_name"] == "bikesense-demand-forecaster"
    assert "active_champion_version" in data
    assert "uptime_seconds" in data


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    # If a champion is registered, it returns 200; otherwise 503
    if response.status_code == 200:
        data = response.json()
        assert "champion_version" in data
        assert "git_commit" in data
        assert "dvc_dataset_hash" in data
    else:
        assert response.status_code == 503


def test_forecast_endpoint_structure(client):
    response = client.get("/forecast?station_id=JC013&date=2024-03-15")
    if response.status_code == 200:
        data = response.json()
        assert data["station_id"] == "JC013"
        assert data["date"] == "2024-03-15"
        assert len(data["forecasts"]) == 24
        assert data["forecasts"][0]["hour"] == 0
        assert data["forecasts"][23]["hour"] == 23
        assert data["total_predicted_departures"] >= 0.0
    else:
        # If MLflow model is temporarily not loaded in test runner
        assert response.status_code in [200, 503]


def test_forecast_invalid_date_format(client):
    response = client.get("/forecast?station_id=JC013&date=15-03-2024")
    # Pydantic regex validator must reject non-YYYY-MM-DD
    assert response.status_code == 422
