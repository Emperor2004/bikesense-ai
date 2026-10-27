"""Promotion gate module for champion/challenger comparison."""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from src.models.metrics import compute_all_metrics


def evaluate_model_on_data(
    model_uri: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> dict[str, float]:
    """Load model from URI and evaluate on dataset."""
    model = mlflow.pyfunc.load_model(model_uri)
    X = df[feature_cols]
    y = df[target_col].to_numpy()
    preds = np.maximum(0.0, model.predict(X))
    return compute_all_metrics(y, preds)


def run_promotion_gate(
    holdout_version: str = "v1",
    auto_approve: bool = False,
    config_path: Path = Path("configs/gate.yaml"),
) -> bool:
    """Run promotion gate comparing @champion and @challenger models."""
    with open(config_path, "r", encoding="utf-8") as f:
        gate_cfg = yaml.safe_load(f)["gate"]
    with open("configs/model.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    model_name = gate_cfg["model_name"]
    improvement_margin = float(gate_cfg["improvement_margin_pct"])
    max_shortfall_increase = float(gate_cfg.get("max_shortfall_increase_pct", 5.0))
    feature_cols = model_cfg["feature_columns"]
    target_col = model_cfg["target_column"]

    mlflow.set_tracking_uri(gate_cfg.get("mlflow_tracking_uri", "sqlite:///mlflow.db"))
    client = MlflowClient()

    model_info = client.get_registered_model(model_name)
    aliases = model_info.aliases

    if "challenger" not in aliases:
        print(f"Error: No @challenger alias found in '{model_name}'.")
        return False

    challenger_ver = aliases["challenger"]

    if "champion" not in aliases:
        print(f"No existing @champion found. Automatically promoting version {challenger_ver} to @champion.")
        client.set_registered_model_alias(model_name, "champion", challenger_ver)
        client.delete_registered_model_alias(model_name, "challenger")
        return True

    champion_ver = aliases["champion"]

    if challenger_ver == champion_ver:
        print(f"Challenger and Champion are the same version ({champion_ver}). Nothing to promote.")
        return False

    print(f"\n=== Running Promotion Gate for '{model_name}' ===")
    print(f"Champion Version:   {champion_ver}")
    print(f"Challenger Version: {challenger_ver}")
    print(f"Required MAE Improvement: >= {improvement_margin}%\n")

    # Load holdout evaluation dataset
    features_path = Path(f"data/processed/features_{holdout_version}.parquet")
    if not features_path.exists():
        features_path = Path("data/processed/features.parquet")
    df = pd.read_parquet(features_path)

    # Use the holdout month for unbiased comparison
    with open("config.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    train_keys = params["dataset_versions"][holdout_version]["train"]
    holdout_month = params["months"][train_keys[-1]]
    df["ym"] = pd.to_datetime(df["datetime_hour"]).dt.strftime("%Y-%m")
    holdout_df = df[df["ym"] == holdout_month].copy()

    champ_uri = f"models:/{model_name}@{champion_ver}"
    chall_uri = f"models:/{model_name}@{challenger_ver}"

    print(f"Evaluating both models on holdout month {holdout_month} ({len(holdout_df)} records)...")
    champ_metrics = evaluate_model_on_data(champ_uri, holdout_df, feature_cols, target_col)
    chall_metrics = evaluate_model_on_data(chall_uri, holdout_df, feature_cols, target_col)

    print("\n--- Gate Metrics Comparison ---")
    print(f"Champion   | MAE: {champ_metrics['mae']:<6.4f} | RMSE: {champ_metrics['rmse']:<6.4f} | Shortfall: {champ_metrics['shortfall']:<8.2f}")
    print(f"Challenger | MAE: {chall_metrics['mae']:<6.4f} | RMSE: {chall_metrics['rmse']:<6.4f} | Shortfall: {chall_metrics['shortfall']:<8.2f}")

    # Compute margin
    mae_diff = champ_metrics["mae"] - chall_metrics["mae"]
    mae_pct_improvement = (mae_diff / champ_metrics["mae"]) * 100.0

    shortfall_diff = chall_metrics["shortfall"] - champ_metrics["shortfall"]
    shortfall_pct_increase = (shortfall_diff / champ_metrics["shortfall"]) * 100.0 if champ_metrics["shortfall"] > 0 else 0.0

    print(f"\nMAE Improvement: {mae_pct_improvement:+.2f}% (Threshold: +{improvement_margin}%)")
    print(f"Shortfall Delta: {shortfall_pct_increase:+.2f}% (Max Allowed: +{max_shortfall_increase}%)")

    # Gate decision
    passed_mae = mae_pct_improvement >= improvement_margin
    passed_shortfall = shortfall_pct_increase <= max_shortfall_increase

    if not passed_mae:
        print(f"\n[REJECTED] Challenger improvement ({mae_pct_improvement:+.2f}%) below required margin (+{improvement_margin}%).")
        return False

    if not passed_shortfall:
        print(f"\n[REJECTED] Challenger shortfall increased by {shortfall_pct_increase:.2f}%, exceeding {max_shortfall_increase}%.")
        return False

    print("\n[GATE PASSED] Challenger meets all quantitative promotion criteria!")

    # Human-in-the-loop approval
    if not auto_approve and gate_cfg.get("require_human_approval", True):
        confirm = input(f"Promote version {challenger_ver} to @champion? [y/N]: ").strip().lower()
        if confirm not in ["y", "yes"]:
            print("Promotion cancelled by user.")
            return False

    # Execute Promotion:
    print(f"Promoting version {challenger_ver} to @champion...")
    # Preserve previous champion alias for instant rollback
    client.set_registered_model_alias(model_name, "previous_champion", champion_ver)
    client.set_registered_model_alias(model_name, "champion", challenger_ver)
    client.delete_registered_model_alias(model_name, "challenger")

    print(f"SUCCESS: Version {challenger_ver} is now @champion! (Previous champion {champion_ver} kept as @previous_champion)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run promotion gate comparison.")
    parser.add_argument("--version", type=str, default="v1", help="Dataset version for holdout evaluation")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve promotion if gate passes")
    args = parser.parse_args()

    run_promotion_gate(holdout_version=args.version, auto_approve=args.auto_approve)


if __name__ == "__main__":
    main()
