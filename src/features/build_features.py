"""Feature engineering module for BikeSense AI."""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_all_weather(weather_dir: Path) -> pd.DataFrame:
    """Load and concatenate all cached monthly weather parquets."""
    weather_files = sorted(weather_dir.glob("weather_*.parquet"))
    if not weather_files:
        raise FileNotFoundError(f"No weather parquet files found in {weather_dir}")
    dfs = [pd.read_parquet(f) for f in weather_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime_hour"]).sort_values("datetime_hour")
    return combined


def merge_data_sources(
    departures: pd.DataFrame, weather: pd.DataFrame, holidays: pd.DataFrame
) -> pd.DataFrame:
    """Join departures with weather and holiday indicators on datetime_hour."""
    departures["datetime_hour"] = pd.to_datetime(departures["datetime_hour"])
    weather["datetime_hour"] = pd.to_datetime(weather["datetime_hour"])
    holidays["datetime_hour"] = pd.to_datetime(holidays["datetime_hour"])

    merged = pd.merge(departures, weather, on="datetime_hour", how="inner")
    merged = pd.merge(merged, holidays, on="datetime_hour", how="inner")
    return merged


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from datetime_hour."""
    df = df.copy()
    dt = df["datetime_hour"].dt
    df["hour"] = dt.hour
    df["day_of_week"] = dt.day_of_week
    df["month"] = dt.month
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-station lag features without data leakage.
    
    Lags:
    - lag_24h: Departures at the same station 24 hours prior.
    - lag_168h: Departures at the same station 168 hours (7 days) prior.
    """
    df = df.copy()
    # Sort strictly by station and time to ensure shift() correctly moves hours
    df = df.sort_values(["station_id", "datetime_hour"]).reset_index(drop=True)

    # Note: Since the grid is fully contiguous with hourly frequency per station,
    # shift(24) and shift(168) correspond exactly to 24 hours and 168 hours.
    df["lag_24h"] = df.groupby("station_id")["departures"].shift(24)
    df["lag_168h"] = df.groupby("station_id")["departures"].shift(168)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-station rolling 7-day same-hour mean demand.
    
    For a given station and hour (e.g. 8 AM), compute the mean departures of the preceding
    7 occurrences of that hour.
    
    Strict leakage safeguard:
    shift(1) ensures the current hour's departure value is never included in its own rolling mean.
    """
    df = df.copy()
    df = df.sort_values(["station_id", "hour", "datetime_hour"]).reset_index(drop=True)

    # Shift by 1 occurrence within the (station_id, hour) group, then roll 7
    shifted = df.groupby(["station_id", "hour"])["departures"].shift(1)
    df["rolling_7d_same_hour_mean"] = (
        shifted.groupby([df["station_id"], df["hour"]])
        .rolling(window=7, min_periods=7)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Re-sort back to standard station and time order
    df = df.sort_values(["station_id", "datetime_hour"]).reset_index(drop=True)
    return df


def build_feature_table(
    departures_path: Path,
    weather_dir: Path,
    holidays_path: Path,
    output_path: Path,
    target_months: list | None = None,
) -> pd.DataFrame:
    """Build the final feature table and save to parquet."""
    print("Reading data sources for feature generation...")
    departures = pd.read_parquet(departures_path)
    weather = load_all_weather(weather_dir)
    holidays = pd.read_parquet(holidays_path)

    merged = merge_data_sources(departures, weather, holidays)
    with_temporal = add_temporal_features(merged)
    with_lags = add_lag_features(with_temporal)
    with_rolling = add_rolling_features(with_lags)

    # Drop warm-up rows where lags or rolling means are NaN (the initial 168 hours)
    clean_df = with_rolling.dropna(subset=["lag_24h", "lag_168h", "rolling_7d_same_hour_mean"]).copy()

    # Filter target months if specified
    if target_months:
        clean_df["year_month"] = clean_df["datetime_hour"].dt.strftime("%Y-%m")
        clean_df = clean_df[clean_df["year_month"].isin(target_months)].drop(columns=["year_month"])

    # Ensure final schema order
    expected_columns = [
        "station_id",
        "datetime_hour",
        "departures",
        "hour",
        "day_of_week",
        "month",
        "is_holiday",
        "temperature_2m",
        "precipitation",
        "wind_speed_10m",
        "lag_24h",
        "lag_168h",
        "rolling_7d_same_hour_mean",
    ]
    clean_df = clean_df[expected_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(output_path, index=False)
    print(f"Saved feature table with {len(clean_df)} records to {output_path}")
    return clean_df


def main():
    parser = argparse.ArgumentParser(description="Build features for BikeSense AI.")
    parser.add_argument("--version", type=str, default=None, help="Dataset version (e.g. v1, v2, v3)")
    args = parser.parse_args()

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    version = args.version or params.get("dataset_version", "v1")
    months_dict = params["months"]

    # Retrieve months for this version (training + replay)
    version_cfg = params["dataset_versions"][version]
    included_month_keys = version_cfg["train"] + version_cfg["replay"]
    included_months = [months_dict[k] for k in included_month_keys]

    departures_path = Path("data/raw/departures_hourly.parquet")
    weather_dir = Path("data/external/weather")
    holidays_path = Path("data/external/holidays.parquet")

    # Save to canonical features.parquet (for DVC pipeline) and versioned parquet
    out_canonical = Path("data/processed/features.parquet")
    out_versioned = Path(f"data/processed/features_{version}.parquet")

    print(f"Building features for version {version} ({len(included_months)} months)...")
    df = build_feature_table(
        departures_path,
        weather_dir,
        holidays_path,
        out_canonical,
        target_months=included_months,
    )
    df.to_parquet(out_versioned, index=False)
    print(f"Also saved versioned features copy to {out_versioned}")


if __name__ == "__main__":
    main()
