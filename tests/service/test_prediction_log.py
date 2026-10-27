"""Unit tests for prediction logger."""

import pandas as pd

from service.app.prediction_log import PredictionLogger


def test_prediction_logger_writes(tmp_path):
    log_file = tmp_path / "test_prediction_log.csv"
    logger = PredictionLogger(log_path=log_file)

    sample_forecasts = [
        {"datetime_hour": "2024-03-01 08:00:00", "predicted": 14.5},
        {"datetime_hour": "2024-03-01 09:00:00", "predicted": 12.0},
    ]

    logger.log_forecast(
        station_id="JC001",
        issued_at="2024-03-01T00:00:00Z",
        forecasts=sample_forecasts,
        model_version="1",
        git_commit="abc1234",
    )

    assert log_file.exists()
    df = pd.read_csv(log_file)
    assert len(df) == 2
    assert list(df.columns) == PredictionLogger.EXPECTED_COLUMNS
    assert df.loc[0, "station_id"] == "JC001"
    assert df.loc[0, "predicted"] == 14.5
    assert df.loc[0, "model_version"] == 1
    assert df.loc[0, "git_commit"] == "abc1234"
