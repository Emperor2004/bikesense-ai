"""Seasonal-naive baseline model for bike-share demand forecasting."""

import numpy as np
import pandas as pd


class SeasonalNaiveBaseline:
    """Predicts hourly departures using demand from the same hour one week prior (lag_168h)."""

    def __init__(self, lag_col: str = "lag_168h"):
        self.lag_col = lag_col

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "SeasonalNaiveBaseline":
        """Fitting is a no-op for a seasonal-naive baseline."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions by pulling the lag_168h column."""
        if self.lag_col not in X.columns:
            raise KeyError(f"Column '{self.lag_col}' required for seasonal-naive prediction.")
        # Fill any remaining NaNs with 0
        preds = X[self.lag_col].fillna(0.0).to_numpy(dtype=float)
        # Demand cannot be negative
        return np.maximum(0.0, preds)
