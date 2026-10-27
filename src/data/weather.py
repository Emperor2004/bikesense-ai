"""Module for fetching and caching Open-Meteo historical weather data."""

from pathlib import Path

import pandas as pd
import requests
import yaml


def fetch_weather_month(
    year_month: str,
    latitude: float,
    longitude: float,
    hourly_vars: list[str],
    timezone: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """Fetch hourly weather for one month from Open-Meteo Historical Weather API.
    
    Caches response as parquet in cache_dir (e.g. data/external/weather/weather_2023-07.parquet).
    If cache file exists, loads from disk without calling API.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"weather_{year_month}.parquet"

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    start_date = f"{year_month}-01"
    month_dt = pd.to_datetime(start_date)
    end_date = (month_dt + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")

    api_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": timezone,
    }

    response = requests.get(api_url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "hourly" not in payload:
        raise ValueError(f"Open-Meteo API response missing 'hourly' field: {payload}")

    hourly_data = payload["hourly"]
    time_series = pd.to_datetime(hourly_data["time"])

    df = pd.DataFrame({"datetime_hour": time_series})
    for var in hourly_vars:
        df[var] = hourly_data.get(var, None)

    df.to_parquet(cache_file, index=False)
    return df


def fetch_all_weather(params: dict) -> None:
    """Fetch and cache weather for all months specified in params.yaml."""
    weather_cfg = params["weather"]
    cache_dir = Path("data/external/weather")

    for month_key in sorted(params["months"].keys(), key=lambda k: int(k.replace("m", ""))):
        year_month = params["months"][month_key]
        print(f"Fetching/loading weather for {year_month}...")
        fetch_weather_month(
            year_month=year_month,
            latitude=weather_cfg["latitude"],
            longitude=weather_cfg["longitude"],
            hourly_vars=weather_cfg["hourly_vars"],
            timezone=weather_cfg["timezone"],
            cache_dir=cache_dir,
        )


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    fetch_all_weather(params)
    print("All weather data cached successfully.")


if __name__ == "__main__":
    main()
