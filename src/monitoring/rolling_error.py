"""Rolling error calculation module evaluating 7-day performance degradation."""

import argparse
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient


def get_champion_reference_mae(
    model_name: str = "bikesense-demand-forecaster",
    tracking_uri: str = "sqlite:///mlflow.db",
) -> float:
    """Retrieve the validation MAE of the current champion from MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        model = client.get_registered_model(model_name)
        champ_ver = model.aliases.get("champion")
        if not champ_ver:
            return 1.0

        mv = client.get_model_version(model_name, champ_ver)
        run = client.get_run(mv.run_id)
        # Look for val_mae in logged metrics
        return float(run.data.metrics.get("val_mae", 1.0))
    except Exception as e:
        print(f"Warning: Could not fetch champion reference MAE from MLflow ({e}). Defaulting to 1.0.")
        return 1.0


def compute_rolling_performance(
    actuals_path: Path = Path("data/predictions/monitored_actuals.parquet"),
    window_days: int = 7,
    reference_mae: float | None = None,
    output_dir: Path = Path("reports/monitoring"),
) -> dict[str, Any]:
    """Calculate rolling 7-day performance metrics and compare against reference MAE."""
    if not actuals_path.exists():
        raise FileNotFoundError(f"Monitored actuals table not found at {actuals_path}")

    df = pd.read_parquet(actuals_path)
    if df.empty:
        raise ValueError("Monitored actuals dataset is empty.")

    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    df["date"] = df["datetime_hour"].dt.date

    # Aggregate daily metrics across all stations
    daily = (
        df.groupby("date")
        .agg(
            total_records=("abs_error", "count"),
            daily_mae=("abs_error", "mean"),
            daily_shortfall=("shortfall", "sum"),
        )
        .reset_index()
    )

    # Compute rolling 7-day window metrics
    daily["rolling_7d_mae"] = (
        daily["daily_mae"].rolling(window=window_days, min_periods=1).mean()
    )
    daily["rolling_7d_shortfall"] = (
        daily["daily_shortfall"].rolling(window=window_days, min_periods=1).mean()
    )

    if reference_mae is None:
        reference_mae = get_champion_reference_mae()

    # Calculate degradation factor
    latest_rolling_mae = float(daily["rolling_7d_mae"].iloc[-1])
    degradation_factor = latest_rolling_mae / reference_mae if reference_mae > 0 else 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "rolling_error_series.csv", index=False)

    summary = {
        "latest_date": str(daily["date"].iloc[-1]),
        "days_monitored": len(daily),
        "latest_daily_mae": round(float(daily["daily_mae"].iloc[-1]), 4),
        "latest_rolling_7d_mae": round(latest_rolling_mae, 4),
        "reference_mae": round(reference_mae, 4),
        "degradation_factor": round(degradation_factor, 3),
        "latest_rolling_7d_shortfall": round(float(daily["rolling_7d_shortfall"].iloc[-1]), 2),
    }

    print("\n=== Rolling 7-Day Performance Summary ===")
    print(f"Monitoring Period:     {daily['date'].iloc[0]} to {daily['date'].iloc[-1]} ({len(daily)} days)")
    print(f"Latest Daily MAE:      {summary['latest_daily_mae']}")
    print(f"Rolling 7-Day MAE:     {summary['latest_rolling_7d_mae']}")
    print(f"Champion Reference MAE:{summary['reference_mae']}")
    print(f"Degradation Factor:    {summary['degradation_factor']}x (Threshold: 1.25x)")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute rolling error and degradation factor.")
    parser.add_argument("--config", type=str, default="configs/monitoring.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["monitoring"]

    compute_rolling_performance(
        actuals_path=Path(cfg["joined_actuals_path"]),
        window_days=cfg.get("rolling_window_days", 7),
        output_dir=Path(cfg["reports_dir"]),
    )


if __name__ == "__main__":
    main()
