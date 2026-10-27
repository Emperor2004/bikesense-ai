"""Delayed actuals join module: merges predictions with ground truth demand 24 hours later."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def join_predictions_with_actuals(
    prediction_log_path: Path = Path("data/predictions/prediction_log.csv"),
    actuals_path: Path = Path("data/raw/departures_hourly.parquet"),
    output_path: Path = Path("data/predictions/monitored_actuals.parquet"),
) -> pd.DataFrame:
    """Join issued forecasts with actual observed departures."""
    if not prediction_log_path.exists():
        raise FileNotFoundError(f"Prediction log not found at {prediction_log_path}")

    if not actuals_path.exists():
        raise FileNotFoundError(f"Actuals departures table not found at {actuals_path}")

    pred_df = pd.read_csv(prediction_log_path)
    actuals_df = pd.read_parquet(actuals_path)

    # Standardize types and column names
    pred_df["datetime_hour"] = pd.to_datetime(pred_df["datetime_hour"])
    pred_df["station_id"] = pred_df["station_id"].astype(str)

    actuals_df["datetime_hour"] = pd.to_datetime(actuals_df["datetime_hour"])
    actuals_df["station_id"] = actuals_df["station_id"].astype(str)

    # Inner join on station and datetime_hour: only keeps hours where actuals have arrived
    joined = pd.merge(
        pred_df,
        actuals_df[["station_id", "datetime_hour", "departures"]],
        on=["station_id", "datetime_hour"],
        how="inner",
    ).rename(columns={"departures": "actual"})

    # Compute operational error metrics per hourly record
    joined["abs_error"] = np.abs(joined["actual"] - joined["predicted"])
    joined["shortfall"] = np.maximum(0.0, joined["actual"] - joined["predicted"])

    # Sort chronologically
    joined = joined.sort_values(["datetime_hour", "station_id"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(output_path, index=False)

    print(f"Joined {len(joined)} prediction records with actual demand.")
    print(f"Monitored actuals saved to: {output_path}")
    return joined


def main():
    parser = argparse.ArgumentParser(description="Join prediction log with actual departures.")
    parser.add_argument("--config", type=str, default="configs/monitoring.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["monitoring"]

    join_predictions_with_actuals(
        prediction_log_path=Path(cfg["prediction_log_path"]),
        actuals_path=Path("data/raw/departures_hourly.parquet"),
        output_path=Path(cfg["joined_actuals_path"]),
    )


if __name__ == "__main__":
    main()
