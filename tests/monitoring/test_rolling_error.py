"""Unit tests for rolling 7-day performance calculation."""

import pandas as pd
import pytest

from src.monitoring.rolling_error import compute_rolling_performance


def test_rolling_performance_calculation(tmp_path):
    actuals_file = tmp_path / "actuals.parquet"
    out_dir = tmp_path / "reports"

    # Create 14 days of hourly data (336 hours)
    dates = pd.date_range("2024-03-01 00:00:00", periods=336, freq="h")
    # Set errors to exactly 2.0 on first 7 days, and 4.0 on second 7 days
    records = []
    for dt in dates:
        err = 2.0 if dt.day <= 7 else 4.0
        records.append(
            {
                "station_id": "st_1",
                "datetime_hour": dt,
                "predicted": 10.0,
                "actual": 10.0 + err,
                "abs_error": err,
                "shortfall": err,
            }
        )

    pd.DataFrame(records).to_parquet(actuals_file, index=False)

    summary = compute_rolling_performance(
        actuals_path=actuals_file,
        window_days=7,
        reference_mae=2.0,
        output_dir=out_dir,
    )

    assert summary["days_monitored"] == 14
    # On day 14, the rolling 7-day window covers only days 8-14, where error is 4.0
    assert summary["latest_rolling_7d_mae"] == pytest.approx(4.0)
    # Degradation factor: 4.0 / 2.0 = 2.0x
    assert summary["degradation_factor"] == pytest.approx(2.0)
