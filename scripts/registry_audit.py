"""Registry audit script: verifies 100% lineage compliance across registered models."""

import sys

import mlflow
from mlflow.tracking import MlflowClient


def audit_registry(
    model_name: str = "bikesense-demand-forecaster",
    tracking_uri: str = "sqlite:///mlflow.db",
) -> bool:
    """Audit all registered model versions for required Git and DVC lineage tags."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    try:
        client.get_registered_model(model_name)
    except (mlflow.exceptions.MlflowException, OSError) as e:
        print(f"Error accessing registered model '{model_name}': {e}")
        return False

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"No versions found for model '{model_name}'.")
        return True

    print(f"\n=== Auditing Lineage Compliance for '{model_name}' ===")
    print(f"Total Registered Versions: {len(versions)}\n")

    required_tags = ["git_commit", "dvc_dataset_hash", "dataset_version"]
    all_compliant = True

    print(f"{'Version':<10} | {'Status':<12} | {'Git Commit':<14} | {'DVC Hash':<14} | {'Dataset'}")
    print("-" * 75)

    for mv in sorted(versions, key=lambda v: int(v.version)):
        v_tags = mv.tags or {}
        missing = [t for t in required_tags if t not in v_tags or not v_tags[t]]

        git_hash = v_tags.get("git_commit", "")[:10] or "MISSING"
        dvc_hash = v_tags.get("dvc_dataset_hash", "")[:10] or "MISSING"
        dataset_v = v_tags.get("dataset_version", "MISSING")

        if missing:
            all_compliant = False
            status = f"FAIL ({', '.join(missing)})"
        else:
            status = "PASS"

        print(f"{mv.version:<10} | {status:<12} | {git_hash:<14} | {dvc_hash:<14} | {dataset_v}")

    print("-" * 75)
    if all_compliant:
        print("\nLineage Audit Result: 100% COMPLIANT. All versions carry complete lineage tags.\n")
        return True
    else:
        print("\nLineage Audit Result: NON-COMPLIANT. Some versions lack required lineage metadata.\n")
        return False


def main():
    compliant = audit_registry()
    sys.exit(0 if compliant else 1)


if __name__ == "__main__":
    main()
