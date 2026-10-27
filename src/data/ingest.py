"""Citi Bike data ingestion and departure aggregation module."""

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml

from src.data.validate import (
    validate_departures_table,
    validate_no_future_dates,
    validate_raw_schema,
)


def download_and_extract_month(
    year_month: str,
    base_url: str,
    file_pattern: str,
    output_dir: Path,
) -> Path:
    """Download a month's zip file from Citi Bike tripdata S3 and extract CSV.
    
    Returns:
        Path to the extracted CSV file.
    """
    yyyymm = year_month.replace("-", "")
    target_dir = output_dir / year_month
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if CSV already extracted
    existing_csvs = list(target_dir.glob("*.csv"))
    if existing_csvs:
        return existing_csvs[0]

    filename = file_pattern.format(yyyymm=yyyymm)
    url = f"{base_url.rstrip('/')}/{filename}"
    print(f"Downloading {url} ...")

    resp = requests.get(url, stream=True, timeout=60)
    if resp.status_code != 200:
        # Fallback patterns if needed
        fallback_filenames = [
            f"{yyyymm}-citibike-tripdata.csv.zip",
            f"{yyyymm}-citibike-tripdata.zip",
            f"JC-{yyyymm}-citibike-tripdata.zip",
        ]
        success = False
        for alt in fallback_filenames:
            alt_url = f"{base_url.rstrip('/')}/{alt}"
            print(f"Trying fallback {alt_url} ...")
            alt_resp = requests.get(alt_url, stream=True, timeout=60)
            if alt_resp.status_code == 200:
                resp = alt_resp
                success = True
                break
        if not success:
            resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(target_dir)

    # Find the extracted CSV
    extracted_csvs = list(target_dir.rglob("*.csv"))
    if not extracted_csvs:
        raise FileNotFoundError(f"No CSV found in extracted archive for {year_month}")

    # Return primary CSV
    # Ignore OS X metadata files if present (__MACOSX)
    valid_csvs = [p for p in extracted_csvs if not p.name.startswith("._") and "__MACOSX" not in str(p)]
    return valid_csvs[0]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case Citi Bike format."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map legacy column names if present
    column_mapping = {
        "starttime": "started_at",
        "stoptime": "ended_at",
        "start_station_id": "start_station_id",
        "start_station_name": "start_station_name",
        "end_station_id": "end_station_id",
        "end_station_name": "end_station_name",
        "usertype": "member_casual",
    }
    df = df.rename(columns=column_mapping)
    return df


def load_and_validate_month_trips(csv_path: Path, year_month: str) -> pd.DataFrame:
    """Read month trip CSV, normalize columns, and validate schema."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = normalize_columns(df)

    # Validate raw schema
    schema_errs = validate_raw_schema(df)
    if schema_errs:
        print(f"Warning for {year_month} schema: {schema_errs}")

    date_errs = validate_no_future_dates(df, year_month)
    if date_errs:
        print(f"Warning for {year_month} dates: {date_errs}")

    return df


def identify_top_stations(
    monthly_dfs: list[pd.DataFrame], top_n: int = 30
) -> list[str]:
    """Identify the top N stations by total departure count across provided datasets."""
    combined_starts = []
    for df in monthly_dfs:
        if "start_station_id" in df.columns:
            valid_stations = df["start_station_id"].dropna().astype(str)
            # Remove any empty or 'nan' strings
            valid_stations = valid_stations[~valid_stations.isin(["", "nan", "None"])]
            combined_starts.append(valid_stations)

    if not combined_starts:
        raise ValueError("No start_station_id data available to select top stations.")

    all_starts = pd.concat(combined_starts, ignore_index=True)
    top_counts = all_starts.value_counts().head(top_n)
    return list(top_counts.index)


def aggregate_monthly_departures(
    df: pd.DataFrame, station_ids: list[str], year_month: str
) -> pd.DataFrame:
    """Aggregate trip departures to station-hourly counts, zero-filling missing hours.
    
    Returns:
        DataFrame with columns: [station_id, datetime_hour, departures]
    """
    df = df.copy()
    if "start_station_id" not in df.columns or "started_at" not in df.columns:
        raise KeyError("Required columns 'start_station_id' or 'started_at' missing from trips.")

    # Convert start_station_id to string and clean
    df["start_station_id"] = df["start_station_id"].astype(str)
    df = df[df["start_station_id"].isin(station_ids)].copy()

    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df = df.dropna(subset=["started_at"])

    # Floor to nearest hour
    df["datetime_hour"] = df["started_at"].dt.floor("h")

    # Aggregate counts
    agg = (
        df.groupby(["start_station_id", "datetime_hour"])
        .size()
        .reset_index(name="departures")
        .rename(columns={"start_station_id": "station_id"})
    )

    # Build full hourly Cartesian index for the month across top stations
    start_dt = pd.to_datetime(f"{year_month}-01 00:00:00")
    end_dt = (start_dt + pd.offsets.MonthEnd(1)).replace(hour=23, minute=0, second=0)
    full_hours = pd.date_range(start=start_dt, end=end_dt, freq="h")

    idx = pd.MultiIndex.from_product(
        [station_ids, full_hours], names=["station_id", "datetime_hour"]
    )
    full_grid = pd.DataFrame(index=idx).reset_index()

    merged = pd.merge(full_grid, agg, on=["station_id", "datetime_hour"], how="left")
    merged["departures"] = merged["departures"].fillna(0).astype(int)

    # Verify table integrity
    errs = validate_departures_table(merged)
    if errs:
        raise ValueError(f"Departure table validation failed for {year_month}: {errs}")

    return merged


def ingest_all(params: dict) -> None:
    """Download, validate, identify top stations, and generate hourly departures."""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    base_url = params["citibike"]["base_url"]
    file_pattern = params["citibike"].get("file_pattern", "JC-{yyyymm}-citibike-tripdata.csv.zip")
    top_n = params["stations"]["top_n"]

    months_dict = params["months"]
    sorted_month_keys = sorted(months_dict.keys(), key=lambda k: int(k.replace("m", "")))
    all_months = [months_dict[k] for k in sorted_month_keys]

    # Download all CSVs
    month_csv_map = {}
    for ym in all_months:
        csv_path = download_and_extract_month(ym, base_url, file_pattern, raw_dir)
        month_csv_map[ym] = csv_path

    # Dataset v1 training months determine top stations
    v1_train_keys = params["dataset_versions"]["v1"]["train"]
    v1_train_months = [months_dict[k] for k in v1_train_keys]

    print(f"Identifying top {top_n} stations from dataset v1 training months ({v1_train_months})...")
    v1_dfs = [
        load_and_validate_month_trips(month_csv_map[ym], ym)
        for ym in v1_train_months
    ]
    top_station_ids = identify_top_stations(v1_dfs, top_n=top_n)

    # Save top station list for downstream modules
    station_file = processed_dir / "station_list.json"
    with open(station_file, "w", encoding="utf-8") as f:
        json.dump(top_station_ids, f, indent=2)
    print(f"Top {len(top_station_ids)} stations saved to {station_file}")

    # Now aggregate departures for all 14 months
    monthly_departures = []
    for ym in all_months:
        print(f"Aggregating departures for {ym}...")
        df_trips = load_and_validate_month_trips(month_csv_map[ym], ym)
        dep_df = aggregate_monthly_departures(df_trips, top_station_ids, ym)
        monthly_departures.append(dep_df)

    final_departures = pd.concat(monthly_departures, ignore_index=True)
    out_path = raw_dir / "departures_hourly.parquet"
    final_departures.to_parquet(out_path, index=False)
    print(f"Saved {len(final_departures)} hourly departure records to {out_path}")


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    ingest_all(params)


if __name__ == "__main__":
    main()
