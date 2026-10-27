"""Evaluation metrics module for BikeSense AI forecasting models."""

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def under_forecast_shortfall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Under-forecast Shortfall.
    
    Formula: sum(max(0, y_true - y_pred))
    Measures total unmet demand units that were under-predicted.
    A high shortfall represents a heightened risk of empty stations.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    diff = y_true_arr - y_pred_arr
    shortfall = np.maximum(0.0, diff)
    return float(np.sum(shortfall))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate all standard and domain metrics."""
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(root_mean_squared_error(y_true, y_pred), 4),
        "shortfall": round(under_forecast_shortfall(y_true, y_pred), 2),
    }
