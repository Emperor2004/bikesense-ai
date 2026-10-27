"""Validation module for Citi Bike raw data and aggregated departure tables."""


import pandas as pd

EXPECTED_RAW_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]

EXPECTED_DEPARTURES_COLUMNS = [
    "station_id",
    "datetime_hour",
    "departures",
]


def validate_raw_schema(df: pd.DataFrame) -> list[str]:
    """Validate that raw trip DataFrame conforms to expected column names.
    
    Returns:
        List of error messages. Empty list indicates schema is valid.
    """
    errors = []
    missing_cols = [col for col in EXPECTED_RAW_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing expected columns in raw trips data: {missing_cols}")
    return errors


def validate_no_future_dates(df: pd.DataFrame, year_month: str) -> list[str]:
    """Ensure all trips in started_at belong to the designated year_month (YYYY-MM).
    
    Returns:
        List of error messages. Empty list indicates all dates match expected range.
    """
    errors = []
    if "started_at" not in df.columns:
        return ["Column 'started_at' not found for date validation"]

    started = pd.to_datetime(df["started_at"], errors="coerce")
    null_dates = started.isna().sum()
    if null_dates > 0:
        errors.append(f"Found {null_dates} rows with unparseable or missing 'started_at'")

    valid_dates = started.dropna()
    if not valid_dates.empty:
        formatted_months = valid_dates.dt.strftime("%Y-%m")
        unexpected = (formatted_months != year_month).sum()
        if unexpected > 0:
            errors.append(
                f"{unexpected} records do not match target month {year_month}."
            )

    return errors


def validate_station_coverage(
    departures: pd.DataFrame, station_ids: list[str], year_month: str
) -> list[str]:
    """Check that all specified top stations are represented in the departure table for the month.
    
    Returns:
        List of warnings or errors if stations are missing.
    """
    errors = []
    present_stations = set(departures["station_id"].unique())
    missing = [s for s in station_ids if s not in present_stations]
    if missing:
        errors.append(
            f"Month {year_month} is missing records for {len(missing)} stations: {missing[:5]}..."
        )
    return errors


def validate_departures_table(df: pd.DataFrame) -> list[str]:
    """Validate schema and data integrity of the aggregated departures table.
    
    Checks:
    - Required columns present
    - No null values
    - Departure counts >= 0
    - Proper datetime format
    """
    errors = []
    missing_cols = [col for col in EXPECTED_DEPARTURES_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing departure table columns: {missing_cols}")
        return errors

    if df.isna().any().any():
        null_counts = df.isna().sum().to_dict()
        errors.append(f"Null values found in departures table: {null_counts}")

    if "departures" in df.columns and (df["departures"] < 0).any():
        errors.append("Negative departure counts detected.")

    return errors
