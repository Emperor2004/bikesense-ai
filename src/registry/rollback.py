"""Rollback module: instant alias rollback to previous champion model."""

import argparse

import mlflow
from mlflow.tracking import MlflowClient


def rollback_champion(
    to_version: str | None = None,
    model_name: str = "bikesense-demand-forecaster",
    tracking_uri: str = "sqlite:///mlflow.db",
) -> str:
    """Revert the @champion alias to a previous version without redeployment.
    
    Returns:
        The version that is now serving as champion.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_info = client.get_registered_model(model_name)
    aliases = model_info.aliases
    current_champion = aliases.get("champion")

    target_version = to_version
    if not target_version:
        target_version = aliases.get("previous_champion")

    if not target_version:
        raise ValueError(
            f"Cannot rollback: No --to-version provided and no '@previous_champion' alias found for '{model_name}'."
        )

    print(f"\n=== Executing Fast Rollback for '{model_name}' ===")
    print(f"Current Champion:  Version {current_champion}")
    print(f"Rolling Back To:   Version {target_version}")

    # Re-point the alias
    client.set_registered_model_alias(model_name, "champion", str(target_version))

    # Keep record: if we rolled back from current_champion, save it
    if current_champion and current_champion != target_version:
        client.set_registered_model_alias(model_name, "rolled_back_from", str(current_champion))

    print(f"SUCCESS: Version {target_version} is now active as @champion!")
    print("Serving service will immediately load this version on next prediction request without restart.\n")
    return str(target_version)


def main():
    parser = argparse.ArgumentParser(description="Rollback champion model alias.")
    parser.add_argument("--to-version", type=str, default=None, help="Specific version to rollback to")
    args = parser.parse_args()

    rollback_champion(to_version=args.to_version)


if __name__ == "__main__":
    main()
