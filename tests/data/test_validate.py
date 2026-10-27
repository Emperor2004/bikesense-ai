"""Unit tests for data validation module."""

import pandas as pd
import pytest

from src.data.validate import (
    EXPECTED_RAW_COLUMNS,
    validate_departures_table,
    validate_no_future_dates,
    validate_raw_schema,
    validate_station_coverage,
)


@pytest.fixture
def valid_raw_df():
    data = {col: [1] for col in EXPECTED_RAW_COLUMNS}
    data["started_at"] = ["2023-07-15 14:30:00"]
    return pd.DataFrame(data)


@pytest.fixture
def valid_departures_df():
    return pd.DataFrame(
        {
            "station_id": ["station_1", "station_2"],
            "datetime_hour": [
                pd.to_datetime("2023-07-01 00:00:00"),
                pd.to_datetime("2023-07-01 01:00:00"),
            ],
            "departures": [5, 0],
        }
    )


def test_raw_schema_valid(valid_raw_df):
    errors = validate_raw_schema(valid_raw_df)
    assert errors == []


def test_raw_schema_missing_column(valid_raw_df):
    invalid_df = valid_raw_df.drop(columns=["start_station_id"])
    errors = validate_raw_schema(invalid_df)
    assert len(errors) == 1
    assert "start_station_id" in errors[0]


def test_no_future_dates_valid(valid_raw_df):
    errors = validate_no_future_dates(valid_raw_df, "2023-07")
    assert errors == []


def test_no_future_dates_violation(valid_raw_df):
    invalid_df = valid_raw_df.copy()
    invalid_df["started_at"] = ["2023-08-01 10:00:00"]
    errors = validate_no_future_dates(invalid_df, "2023-07")
    assert len(errors) == 1
    assert "2023-07" in errors[0]


def test_departures_no_nulls(valid_departures_df):
    errors = validate_departures_table(valid_departures_df)
    assert errors == []

    null_df = valid_departures_df.copy()
    null_df.loc[0, "departures"] = None
    errors_with_null = validate_departures_table(null_df)
    assert any("Null values found" in err for err in errors_with_null)


def test_departures_non_negative(valid_departures_df):
    negative_df = valid_departures_df.copy()
    negative_df.loc[0, "departures"] = -1
    errors = validate_departures_table(negative_df)
    assert any("Negative departure counts" in err for err in errors)


def test_station_coverage_all_present(valid_departures_df):
    target_stations = ["station_1", "station_2"]
    errors = validate_station_coverage(valid_departures_df, target_stations, "2023-07")
    assert errors == []

    missing_target = ["station_1", "station_missing"]
    errors_missing = validate_station_coverage(valid_departures_df, missing_target, "2023-07")
    assert len(errors_missing) == 1
    assert "station_missing" in errors_missing[0]
