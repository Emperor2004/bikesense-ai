"""Registration module: register best training run in MLflow Model Registry."""

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def register_best_model(
    run_id: str | None = None,
    model_name: str = "bikesense-demand-forecaster",
    as_champion: bool = False,
) -> str:
    """Register run into MLflow Model Registry with tags and alias.
    
    Returns:
        The registered model version string.
    """
    summary_path = Path("reports/evaluation/latest_training_summary.json")

    summary_data = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

    if not run_id:
        if "best_candidate" in summary_data:
            run_id = summary_data["best_candidate"]["run_id"]
        else:
            raise ValueError("No run_id provided and latest_training_summary.json not found.")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    # Get run details
    run = client.get_run(run_id)
    tags = run.data.tags

    model_source = f"runs:/{run_id}/model"
    print(f"Registering model from {model_source} into '{model_name}'...")

    # Register the model
    mv = mlflow.register_model(model_source, model_name)
    version = mv.version

    # Carry forward lineage tags
    git_commit = tags.get("git_commit", summary_data.get("git_commit", "unknown"))
    dvc_hash = tags.get("dvc_dataset_hash", summary_data.get("dvc_dataset_hash", "unknown"))
    dataset_version = tags.get("dataset_version", summary_data.get("dataset_version", "v1"))

    client.set_model_version_tag(model_name, version, "git_commit", git_commit)
    client.set_model_version_tag(model_name, version, "dvc_dataset_hash", dvc_hash)
    client.set_model_version_tag(model_name, version, "dataset_version", dataset_version)
    client.set_model_version_tag(model_name, version, "run_id", run_id)

    # Alias assignment:
    # If explicitly requested or if no champion exists yet, set as @champion
    existing_aliases = client.get_registered_model(model_name).aliases

    if as_champion or "champion" not in existing_aliases:
        print(f"Setting version {version} as @champion...")
        client.set_registered_model_alias(model_name, "champion", version)
    else:
        print(f"Setting version {version} as @challenger...")
        client.set_registered_model_alias(model_name, "challenger", version)

    print(f"Model version {version} successfully registered in '{model_name}' with full lineage tags.")
    return version


def main():
    parser = argparse.ArgumentParser(description="Register candidate model in MLflow Registry.")
    parser.add_argument("--run-id", type=str, default=None, help="MLflow run ID to register")
    parser.add_argument("--as-champion", action="store_true", help="Directly set as champion")
    args = parser.parse_args()

    register_best_model(run_id=args.run_id, as_champion=args.as_champion)


if __name__ == "__main__":
    main()
