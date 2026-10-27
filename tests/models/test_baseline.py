"""Unit tests for the seasonal-naive baseline model."""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import SeasonalNaiveBaseline


def test_seasonal_naive_prediction():
    df = pd.DataFrame(
        {
            "station_id": ["st_1", "st_2", "st_3"],
            "lag_168h": [12.0, 0.0, np.nan],
        }
    )
    model = SeasonalNaiveBaseline()
    model.fit(df)
    preds = model.predict(df)

    assert len(preds) == 3
    assert preds[0] == 12.0
    assert preds[1] == 0.0
    assert preds[2] == 0.0  # NaN filled with 0.0


def test_seasonal_naive_missing_column():
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    model = SeasonalNaiveBaseline()
    with pytest.raises(KeyError, match="lag_168h"):
        model.predict(df)
