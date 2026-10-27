"""Replay driver module to simulate production time."""

import argparse
from pathlib import Path

import pandas as pd
import requests
import yaml


def simulate_or_request_forecast(
    station_id: str,
    date_str: str,
    service_url: str = "http://localhost:8000",
) -> list[dict]:
    """Request a 24-hour forecast from the service or fallback to simulated output if offline."""
    endpoint = f"{service_url.rstrip('/')}/forecast"
    params = {"station_id": station_id, "date": date_str}

    try:
        resp = requests.get(endpoint, params=params, timeout=5)
        if resp.status_code == 200:
            payload = resp.json()
            return payload.get("forecasts", [])
    except requests.RequestException:
        # Service unavailable or offline; proceed to fallback
        pass

    # Offline/fallback simulation for testing prior to serving container start:
    # Generates a baseline prediction placeholder with standard diurnal curve
    predictions = []
    for h in range(24):
        # Synthetic departure curve peaking during morning/evening commute
        pred_val = 2.0 + 8.0 * (1.0 if h in [8, 9, 17, 18] else 0.2)
        predictions.append(
            {
                "hour": h,
                "predicted_departures": round(pred_val, 2),
            }
        )
    return predictions


def run_replay(
    dataset_version: str,
    service_url: str = "http://localhost:8000",
) -> pd.DataFrame:
    """Run replay simulation across the designated replay window."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    months_dict = params["months"]
    replay_keys = params["dataset_versions"][dataset_version]["replay"]
    replay_months = [months_dict[k] for k in replay_keys]

    # Load ground truth features
    features_path = Path(f"data/processed/features_{dataset_version}.parquet")
    if not features_path.exists():
        features_path = Path("data/processed/features.parquet")
    if not features_path.exists():
        raise FileNotFoundError("Feature table not found. Run feature pipeline first.")

    features = pd.read_parquet(features_path)
    features["year_month"] = features["datetime_hour"].dt.strftime("%Y-%m")
    replay_df = features[features["year_month"].isin(replay_months)].copy()

    stations = sorted(replay_df["station_id"].unique())
    # Generate list of dates to replay
    min_date = replay_df["datetime_hour"].dt.date.min()
    max_date = replay_df["datetime_hour"].dt.date.max()
    date_range = pd.date_range(start=min_date, end=max_date, freq="D")

    pred_records = []
    print(f"Replaying {len(date_range)} days for {len(stations)} stations ({dataset_version})...")

    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    log_csv = pred_dir / "prediction_log.csv"

    for cur_date in date_range:
        date_str = cur_date.strftime("%Y-%m-%d")
        issue_time_str = f"{date_str} 00:00:00"

        for st in stations:
            forecasts = simulate_or_request_forecast(st, date_str, service_url=service_url)
            for item in forecasts:
                hour = item["hour"]
                pred = item["predicted_departures"]
                dt_hour = pd.to_datetime(f"{date_str} {hour:02d}:00:00")
                pred_records.append(
                    {
                        "station_id": st,
                        "datetime_hour": dt_hour,
                        "predicted": pred,
                        "issued_at": issue_time_str,
                    }
                )

    pred_df = pd.DataFrame(pred_records)

    # Append to prediction log
    header = not log_csv.exists()
    pred_df.to_csv(log_csv, mode="a", index=False, header=header)
    print(f"Logged {len(pred_df)} predictions to {log_csv}")

    # Join with ground truth actual demand
    joined = pd.merge(
        pred_df,
        replay_df[["station_id", "datetime_hour", "departures"]],
        on=["station_id", "datetime_hour"],
        how="inner",
    ).rename(columns={"departures": "actual"})

    out_replay = pred_dir / f"replay_{dataset_version}.parquet"
    joined.to_parquet(out_replay, index=False)
    print(f"Replay actuals joined dataset saved to {out_replay}")
    return joined


def main():
    parser = argparse.ArgumentParser(description="Replay simulated production window.")
    parser.add_argument("--version", type=str, default="v1", help="Dataset version (v1, v2, v3)")
    parser.add_argument("--service-url", type=str, default="http://localhost:8000", help="FastAPI service URL")
    args = parser.parse_args()

    run_replay(dataset_version=args.version, service_url=args.service_url)


if __name__ == "__main__":
    main()
