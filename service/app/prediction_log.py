"""Thread-safe prediction logging module for BikeSense AI serving service."""

import threading
from pathlib import Path
from typing import ClassVar

import pandas as pd


class PredictionLogger:
    """Appends issued hourly predictions to persistent CSV log."""

    EXPECTED_COLUMNS: ClassVar[list[str]] = [
        "station_id",
        "datetime_hour",
        "predicted",
        "issued_at",
        "model_version",
        "git_commit",
    ]

    def __init__(self, log_path: Path = Path("data/predictions/prediction_log.csv")):
        self.log_path = log_path
        self._lock = threading.Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_forecast(
        self,
        station_id: str,
        issued_at: str,
        forecasts: list[dict],
        model_version: str,
        git_commit: str,
    ) -> None:
        """Atomically append hourly forecast records to the prediction log."""
        records = []
        for item in forecasts:
            dt_hour = item["datetime_hour"]
            pred = item["predicted"]
            records.append(
                {
                    "station_id": str(station_id),
                    "datetime_hour": pd.to_datetime(dt_hour),
                    "predicted": float(pred),
                    "issued_at": issued_at,
                    "model_version": str(model_version),
                    "git_commit": str(git_commit),
                }
            )

        df = pd.DataFrame(records)

        with self._lock:
            write_header = not self.log_path.exists()
            df.to_csv(self.log_path, mode="a", index=False, header=write_header)
