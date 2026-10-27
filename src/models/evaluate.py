"""Evaluation module comparing models and generating reports."""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.models.baseline import SeasonalNaiveBaseline
from src.models.metrics import compute_all_metrics


def evaluate_window(
    dataset_version: str = "v1",
    model_alias: str = "champion",
    output_dir: Path = Path("reports/evaluation"),
) -> pd.DataFrame:
    """Evaluate models on the replay window and generate markdown reports."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    with open("configs/model.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    feature_cols = model_cfg["feature_columns"]
    target_col = model_cfg["target_column"]

    months_dict = params["months"]
    replay_keys = params["dataset_versions"][dataset_version]["replay"]
    replay_months = [months_dict[k] for k in replay_keys]

    features_path = Path(f"data/processed/features_{dataset_version}.parquet")
    if not features_path.exists():
        features_path = Path("data/processed/features.parquet")
    df = pd.read_parquet(features_path)

    df["ym"] = pd.to_datetime(df["datetime_hour"]).dt.strftime("%Y-%m")
    eval_df = df[df["ym"].isin(replay_months)].copy()

    if eval_df.empty:
        raise ValueError(f"No records found for replay months: {replay_months}")

    X_eval = eval_df[feature_cols]
    y_eval = eval_df[target_col].to_numpy()

    # 1. Baseline evaluation
    baseline = SeasonalNaiveBaseline()
    baseline_preds = baseline.predict(eval_df)
    baseline_metrics = compute_all_metrics(y_eval, baseline_preds)

    # 2. Champion evaluation from MLflow Registry
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = f"models:/bikesense-demand-forecaster@{model_alias}"

    try:
        champion_model = mlflow.pyfunc.load_model(model_uri)
        champ_preds = np.maximum(0.0, champion_model.predict(X_eval))
        champ_metrics = compute_all_metrics(y_eval, champ_preds)
    except (mlflow.exceptions.MlflowException, OSError) as e:
        print(f"Could not load {model_uri}: {e}. Evaluating baseline only.")
        champ_metrics = {"mae": np.nan, "rmse": np.nan, "shortfall": np.nan}

    # Summary table
    records = [
        {
            "Model": "Seasonal-Naive Baseline (168h)",
            "MAE": baseline_metrics["mae"],
            "RMSE": baseline_metrics["rmse"],
            "Shortfall (Units)": baseline_metrics["shortfall"],
        },
        {
            "Model": f"MLflow @{model_alias}",
            "MAE": champ_metrics["mae"],
            "RMSE": champ_metrics["rmse"],
            "Shortfall (Units)": champ_metrics["shortfall"],
        },
    ]
    summary_df = pd.DataFrame(records)

    # Markdown report output
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"evaluation_{dataset_version}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Report — Dataset Version {dataset_version}\n\n")
        f.write(f"**Evaluation Window (Replay):** {', '.join(replay_months)}\n")
        f.write(f"**Total Hourly Records Evaluated:** {len(eval_df):,}\n\n")
        f.write("## Overall Window Performance\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n---\n")

    print(f"Evaluation report generated at: {report_file}")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate forecasting models.")
    parser.add_argument("--version", type=str, default="v1", help="Dataset version (v1, v2, v3)")
    parser.add_argument("--alias", type=str, default="champion", help="Model alias in MLflow registry")
    args = parser.parse_args()
    evaluate_window(dataset_version=args.version, model_alias=args.alias)


if __name__ == "__main__":
    main()
