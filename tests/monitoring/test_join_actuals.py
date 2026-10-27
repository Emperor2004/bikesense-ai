"""Unit tests for delayed actuals join module."""

import pandas as pd

from src.monitoring.join_actuals import join_predictions_with_actuals


def test_actuals_join_matching(tmp_path):
    pred_path = tmp_path / "pred.csv"
    actuals_path = tmp_path / "actuals.parquet"
    out_path = tmp_path / "joined.parquet"

    # Prediction log has 3 hours
    pred_df = pd.DataFrame(
        {
            "station_id": ["st_1", "st_1", "st_1"],
            "datetime_hour": [
                "2024-03-01 00:00:00",
                "2024-03-01 01:00:00",
                "2024-03-01 02:00:00",
            ],
            "predicted": [5.0, 10.0, 8.0],
            "issued_at": ["2024-03-01T00:00:00Z"] * 3,
            "model_version": ["1"] * 3,
            "git_commit": ["test"] * 3,
        }
    )
    pred_df.to_csv(pred_path, index=False)

    # Actuals only arrived for the first 2 hours (hour 2 is delayed / not arrived)
    actuals_df = pd.DataFrame(
        {
            "station_id": ["st_1", "st_1"],
            "datetime_hour": [
                pd.to_datetime("2024-03-01 00:00:00"),
                pd.to_datetime("2024-03-01 01:00:00"),
            ],
            "departures": [7, 6],
        }
    )
    actuals_df.to_parquet(actuals_path, index=False)

    joined = join_predictions_with_actuals(
        prediction_log_path=pred_path,
        actuals_path=actuals_path,
        output_path=out_path,
    )

    # Hour 2 must not be present since actuals have not arrived
    assert len(joined) == 2
    # Hour 0: pred=5.0, actual=7 -> abs_error=2.0, shortfall=2.0
    assert joined.loc[0, "abs_error"] == 2.0
    assert joined.loc[0, "shortfall"] == 2.0
    # Hour 1: pred=10.0, actual=6 -> abs_error=4.0, shortfall=0.0
    assert joined.loc[1, "abs_error"] == 4.0
    assert joined.loc[1, "shortfall"] == 0.0
