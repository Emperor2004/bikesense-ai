"""Unit tests for evaluation metrics (MAE, RMSE, Under-forecast Shortfall)."""

import numpy as np
import pytest

from src.models.metrics import (
    compute_all_metrics,
    mean_absolute_error,
    root_mean_squared_error,
    under_forecast_shortfall,
)


def test_mae_calculation():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 35.0])
    # Errors: |10-12|=2, |20-18|=2, |30-35|=5 -> mean = 9/3 = 3.0
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(3.0)


def test_rmse_calculation():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([13.0, 16.0])
    # Squared errors: 3^2=9, 4^2=16 -> mean=12.5 -> sqrt(12.5) ~= 3.5355
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(12.5))


def test_shortfall_under_prediction():
    y_true = np.array([10.0, 25.0, 15.0])
    y_pred = np.array([5.0, 20.0, 15.0])
    # Under-predicted: (10-5)=5, (25-20)=5, (15-15)=0 -> Total = 10.0
    assert under_forecast_shortfall(y_true, y_pred) == pytest.approx(10.0)


def test_shortfall_over_prediction():
    y_true = np.array([10.0, 15.0])
    y_pred = np.array([15.0, 20.0])
    # All predictions are greater than actuals -> shortfall must be 0
    assert under_forecast_shortfall(y_true, y_pred) == pytest.approx(0.0)


def test_compute_all_metrics():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([10.0, 20.0])
    metrics = compute_all_metrics(y_true, y_pred)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["shortfall"] == 0.0
