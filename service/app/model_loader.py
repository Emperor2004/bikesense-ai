"""Dynamic model resolver and loader for MLflow Model Registry champion alias."""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


class DynamicModelLoader:
    """Manages cached model instance, dynamically refreshing when the @champion alias moves."""

    def __init__(
        self,
        model_name: str = "bikesense-demand-forecaster",
        tracking_uri: str | None = None,
    ):
        self.model_name = model_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()
        self.cached_model: Any = None
        self.current_version: str | None = None
        self.metadata: dict[str, str] = {}

    def get_champion_version(self) -> str | None:
        """Query MLflow registry to determine current version aliased to @champion."""
        try:
            model = self.client.get_registered_model(self.model_name)
            return model.aliases.get("champion")
        except Exception:
            return None

    def load_champion_model(self) -> tuple[Any, dict[str, str]]:
        """Load or refresh the active @champion model from registry.
        
        Returns:
            Tuple of (model_object, metadata_dict)
        """
        active_version = self.get_champion_version()

        if not active_version:
            raise RuntimeError(
                f"No active '@champion' alias found for registered model '{self.model_name}'."
            )

        # Check if version has changed or model is not yet loaded
        if self.cached_model is None or self.current_version != active_version:
            print(f"[ModelLoader] Loading new champion model: version {active_version}...")
            # If active_version is numeric, load via models:/<name>/<version>
            if str(active_version).isdigit():
                model_uri = f"models:/{self.model_name}/{active_version}"
            else:
                model_uri = f"models:/{self.model_name}@{active_version}"

            try:
                loaded_model = mlflow.pyfunc.load_model(model_uri)
            except Exception as e:
                # Cross-platform fallback: resolve via model source if URI points to host paths
                mv = self.client.get_model_version(self.model_name, active_version)
                source = getattr(mv, "source", "")
                recovered = False
                if source.startswith("models:/m-"):
                    model_id = source.replace("models:/", "")
                    for candidate in [
                        Path(f"/app/mlruns/1/models/{model_id}/artifacts"),
                        Path(f"mlruns/1/models/{model_id}/artifacts"),
                    ]:
                        if candidate.exists():
                            loaded_model = mlflow.pyfunc.load_model(str(candidate))
                            recovered = True
                            break
                if not recovered:
                    raise e

            # Retrieve model version tags for lineage
            mv = self.client.get_model_version(self.model_name, active_version)
            tags = mv.tags or {}

            self.cached_model = loaded_model
            self.current_version = active_version
            self.metadata = {
                "model_name": self.model_name,
                "champion_version": str(active_version),
                "run_id": mv.run_id or "unknown",
                "git_commit": tags.get("git_commit", "unknown"),
                "dvc_dataset_hash": tags.get("dvc_dataset_hash", "unknown"),
                "dataset_version": tags.get("dataset_version", "unknown"),
                "loaded_at": datetime.now(UTC).isoformat(),
            }
            print(f"[ModelLoader] Champion version {active_version} loaded successfully.")

        return self.cached_model, self.metadata
