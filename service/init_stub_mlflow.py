"""Helper script to initialize an empty MLflow SQLite registry database for container builds."""

import os
from pathlib import Path


def init_stub():
    db_path = Path("mlflow.db")
    if db_path.exists():
        print(f"[init_stub_mlflow] Existing {db_path} found. Keeping current database.")
        return

    print("[init_stub_mlflow] Initializing clean MLflow registry SQLite database...")
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    try:
        client.create_registered_model("bikesense-demand-forecaster")
        print("[init_stub_mlflow] Registered model 'bikesense-demand-forecaster' created.")
    except Exception as e:
        print(f"[init_stub_mlflow] Registered model note: {e}")

    Path("mlruns/0").mkdir(parents=True, exist_ok=True)
    meta_path = Path("mlruns/0/meta.yaml")
    if not meta_path.exists():
        meta_path.write_text(
            "artifact_location: file:///app/mlruns/0\n"
            "experiment_id: '0'\n"
            "lifecycle_stage: active\n"
            "name: Default\n",
            encoding="utf-8",
        )
    print("[init_stub_mlflow] MLflow stub database and mlruns/ directory ready.")


if __name__ == "__main__":
    init_stub()
