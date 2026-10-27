"""Model training and MLflow tracking pipeline for BikeSense AI."""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.baseline import SeasonalNaiveBaseline
from src.models.metrics import compute_all_metrics


def get_git_commit() -> str:
    """Retrieve the current Git commit hash."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return commit
    except (subprocess.SubprocessError, OSError):
        return "unknown_commit"


def get_dvc_dataset_hash(stage_name: str = "features") -> str:
    """Extract the content hash of the output feature parquet from dvc.lock."""
    lock_path = Path("dvc.lock")
    if not lock_path.exists():
        return "no_dvc_lock"
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            lock = yaml.safe_load(f)
        stage = lock.get("stages", {}).get(stage_name, {})
        outs = stage.get("outs", [])
        if outs and "hash" in outs[0]:
            return outs[0]["hash"]
        if outs and "md5" in outs[0]:
            return outs[0]["md5"]
        return "unknown_dvc_hash"
    except (KeyError, yaml.YAMLError, OSError):
        return "unknown_dvc_hash"


def time_ordered_train_val_split(
    df: pd.DataFrame, train_months: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a strict, time-ordered train/val split without future leakage.
    
    The last month in train_months is reserved as the validation set.
    """
    df = df.copy()
    df["ym"] = pd.to_datetime(df["datetime_hour"]).dt.strftime("%Y-%m")

    # Filter only to data within train_months
    filtered = df[df["ym"].isin(train_months)].sort_values("datetime_hour").reset_index(drop=True)

    val_month = train_months[-1]
    fit_months = train_months[:-1]

    train_df = filtered[filtered["ym"].isin(fit_months)].copy()
    val_df = filtered[filtered["ym"] == val_month].copy()

    # Verify no overlap or reverse ordering
    if not train_df.empty and not val_df.empty:
        max_train_time = train_df["datetime_hour"].max()
        min_val_time = val_df["datetime_hour"].min()
        if max_train_time >= min_val_time:
            raise ValueError(
                f"Data leakage detected: max_train_time ({max_train_time}) >= min_val_time ({min_val_time})"
            )

    return train_df, val_df


def train_and_log_models(dataset_version: str | None = None) -> dict[str, Any]:
    """Train Baseline, Ridge, and LightGBM models, logging runs to MLflow."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if dataset_version is None:
        dataset_version = params.get("dataset_version", "v1")
    with open("configs/model.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    feature_cols = model_cfg["feature_columns"]
    target_col = model_cfg["target_column"]

    months_dict = params["months"]
    train_keys = params["dataset_versions"][dataset_version]["train"]
    train_months = [months_dict[k] for k in train_keys]

    features_path = Path(f"data/processed/features_{dataset_version}.parquet")
    if not features_path.exists():
        features_path = Path("data/processed/features.parquet")
    df = pd.read_parquet(features_path)

    train_df, val_df = time_ordered_train_val_split(df, train_months)
    print(f"Dataset {dataset_version} split: Train records = {len(train_df)}, Val records = {len(val_df)}")

    X_train, y_train = train_df[feature_cols], train_df[target_col].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df[target_col].to_numpy()

    git_commit = get_git_commit()
    dvc_hash = get_dvc_dataset_hash()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bikesense-demand-forecasting")

    results = []

    # 1. Seasonal-Naive Baseline
    print("Training 1/3: Seasonal-Naive Baseline...")
    with mlflow.start_run(run_name=f"baseline_naive_{dataset_version}"):
        baseline = SeasonalNaiveBaseline()
        baseline.fit(X_train, y_train)

        val_preds = baseline.predict(val_df)
        train_preds = baseline.predict(train_df)

        train_metrics = compute_all_metrics(y_train, train_preds)
        val_metrics = compute_all_metrics(y_val, val_preds)

        mlflow.log_param("model_type", "seasonal_naive")
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.set_tags(
            {
                "git_commit": git_commit,
                "dvc_dataset_hash": dvc_hash,
                "dataset_version": dataset_version,
            }
        )
        results.append(
            {
                "model_name": "seasonal_naive",
                "val_mae": val_metrics["mae"],
                "val_shortfall": val_metrics["shortfall"],
                "run_id": mlflow.active_run().info.run_id,
            }
        )

    # 2. Ridge Regression Baseline
    print("Training 2/3: Ridge Regression...")
    with mlflow.start_run(run_name=f"ridge_{dataset_version}"):
        alpha = model_cfg["ridge"]["alpha"]
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        pipe.fit(X_train, y_train)

        train_preds = np.maximum(0.0, pipe.predict(X_train))
        val_preds = np.maximum(0.0, pipe.predict(X_val))

        train_metrics = compute_all_metrics(y_train, train_preds)
        val_metrics = compute_all_metrics(y_val, val_preds)

        mlflow.log_param("model_type", "ridge")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.set_tags(
            {
                "git_commit": git_commit,
                "dvc_dataset_hash": dvc_hash,
                "dataset_version": dataset_version,
            }
        )
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        results.append(
            {
                "model_name": "ridge",
                "val_mae": val_metrics["mae"],
                "val_shortfall": val_metrics["shortfall"],
                "run_id": mlflow.active_run().info.run_id,
            }
        )

    # 3. LightGBM Candidates (Preset experiments)
    lgbm_experiments = model_cfg.get("lightgbm", {}).get("experiments", [])
    for i, exp in enumerate(lgbm_experiments, start=1):
        name = exp["name"]
        print(f"Training LightGBM Candidate {i}/{len(lgbm_experiments)}: {name}...")
        with mlflow.start_run(run_name=f"{name}_{dataset_version}"):
            params_to_fit = {k: v for k, v in exp.items() if k != "name"}
            reg = lgb.LGBMRegressor(**params_to_fit, objective="regression", verbose=-1)
            reg.fit(X_train, y_train)

            train_preds = np.maximum(0.0, reg.predict(X_train))
            val_preds = np.maximum(0.0, reg.predict(X_val))

            train_metrics = compute_all_metrics(y_train, train_preds)
            val_metrics = compute_all_metrics(y_val, val_preds)

            mlflow.log_param("model_type", "lightgbm")
            mlflow.log_param("candidate_name", name)
            mlflow.log_param("is_tuned_trial", False)
            mlflow.log_params(params_to_fit)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.set_tags(
                {
                    "git_commit": git_commit,
                    "dvc_dataset_hash": dvc_hash,
                    "dataset_version": dataset_version,
                }
            )
            mlflow.lightgbm.log_model(reg, artifact_path="model")

            results.append(
                {
                    "model_name": name,
                    "val_mae": val_metrics["mae"],
                    "val_shortfall": val_metrics["shortfall"],
                    "run_id": mlflow.active_run().info.run_id,
                }
            )

    # 4. Hyperparameter Tuning Exploration
    tuning_cfg = model_cfg.get("tuning", {})
    if tuning_cfg.get("enabled", False):
        from sklearn.model_selection import ParameterSampler

        param_dist = tuning_cfg.get("param_distributions", {})
        n_trials = tuning_cfg.get("n_trials", 6)
        rng_seed = tuning_cfg.get("random_state", 42)
        sampler = list(ParameterSampler(param_dist, n_iter=n_trials, random_state=rng_seed))

        print(f"\nRunning Hyperparameter Tuning ({len(sampler)} trials)...")
        for idx, trial_params in enumerate(sampler, start=1):
            trial_name = f"lgbm_tune_trial_{idx}"
            print(f"Trial {idx}/{len(sampler)}: {trial_params}")
            with mlflow.start_run(run_name=f"{trial_name}_{dataset_version}"):
                reg = lgb.LGBMRegressor(
                    **trial_params,
                    objective="regression",
                    random_state=rng_seed,
                    verbose=-1,
                )
                reg.fit(X_train, y_train)

                train_preds = np.maximum(0.0, reg.predict(X_train))
                val_preds = np.maximum(0.0, reg.predict(X_val))

                train_metrics = compute_all_metrics(y_train, train_preds)
                val_metrics = compute_all_metrics(y_val, val_preds)

                mlflow.log_param("model_type", "lightgbm")
                mlflow.log_param("candidate_name", trial_name)
                mlflow.log_param("is_tuned_trial", True)
                mlflow.log_params(trial_params)
                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
                mlflow.set_tags(
                    {
                        "git_commit": git_commit,
                        "dvc_dataset_hash": dvc_hash,
                        "dataset_version": dataset_version,
                        "tuning_trial": str(idx),
                    }
                )
                mlflow.lightgbm.log_model(reg, artifact_path="model")

                results.append(
                    {
                        "model_name": trial_name,
                        "val_mae": val_metrics["mae"],
                        "val_shortfall": val_metrics["shortfall"],
                        "run_id": mlflow.active_run().info.run_id,
                        "params": trial_params,
                    }
                )

    # Rank and pick best candidate
    best_candidate = min(results, key=lambda r: r["val_mae"])
    print(
        f"\nTraining Complete! Best model: {best_candidate['model_name']} "
        f"(Val MAE: {best_candidate['val_mae']}, Shortfall: {best_candidate['val_shortfall']})"
    )

    out_summary = Path("reports/evaluation")
    out_summary.mkdir(parents=True, exist_ok=True)
    summary_file = out_summary / "latest_training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_candidate": best_candidate,
                "all_runs": results,
                "dataset_version": dataset_version,
                "git_commit": git_commit,
                "dvc_dataset_hash": dvc_hash,
            },
            f,
            indent=2,
        )

    return best_candidate


def main():
    parser = argparse.ArgumentParser(description="Train demand forecasting models.")
    parser.add_argument("--version", type=str, default=None, help="Dataset version (defaults to config.yaml)")
    args = parser.parse_args()
    train_and_log_models(dataset_version=args.version)


if __name__ == "__main__":
    main()
