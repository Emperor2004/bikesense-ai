"""Evidently AI data and prediction drift report generator."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def generate_drift_report(
    reference_path: Path,
    current_path: Path,
    output_dir: Path = Path("reports/monitoring"),
    version_label: str = "v1",
    drift_share_threshold: float = 0.30,
) -> dict[str, Any]:
    """Generate Evidently HTML drift report and summary JSON comparing reference and current data."""
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference dataset not found at {reference_path}")
    if not current_path.exists():
        raise FileNotFoundError(f"Current evaluation dataset not found at {current_path}")

    print(f"[DriftReport] Loading reference data from {reference_path}...")
    ref_df = pd.read_parquet(reference_path)

    print(f"[DriftReport] Loading current data from {current_path}...")
    curr_df = pd.read_parquet(current_path)

    with open("configs/monitoring.yaml", "r", encoding="utf-8") as f:
        mon_cfg = yaml.safe_load(f)["monitoring"]

    features_to_check = (
        mon_cfg["features_to_monitor"]["numerical"]
        + mon_cfg["features_to_monitor"]["categorical"]
    )

    # Filter to selected features present in both dataframes
    common_cols = [c for c in features_to_check if c in ref_df.columns and c in curr_df.columns]
    
    # If target / prediction exists, include it for target drift
    target_col = "departures" if "departures" in ref_df.columns and "departures" in curr_df.columns else None
    if target_col and target_col not in common_cols:
        common_cols.append(target_col)

    ref_subset = ref_df[common_cols].copy()
    curr_subset = curr_df[common_cols].copy()

    # Build Evidently Report with DataDriftPreset
    print(f"[DriftReport] Running Evidently DataDriftPreset across {len(common_cols)} features...")
    report = Report(metrics=[DataDriftPreset(drift_share=drift_share_threshold)])
    report.run(reference_data=ref_subset, current_data=curr_subset)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"drift_report_{version_label}.html"
    json_path = output_dir / f"drift_summary_{version_label}.json"

    # Save interactive HTML dashboard
    report.save_html(str(html_path))

    # Extract structured dictionary
    report_dict = report.as_dict()

    # Parse key summary metrics from Evidently output structure
    drift_metrics = report_dict.get("metrics", [{}])[0].get("result", {})
    drift_share = drift_metrics.get("drift_share", 0.0)
    number_of_drifted = drift_metrics.get("number_of_drifted_columns", 0)
    total_columns = drift_metrics.get("number_of_columns", len(common_cols))
    dataset_drift = drift_metrics.get("dataset_drift", drift_share > drift_share_threshold)

    drifted_features = []
    drift_by_column = drift_metrics.get("drift_by_columns", {})
    for col_name, col_stats in drift_by_column.items():
        if col_stats.get("drift_detected", False):
            drifted_features.append(
                {
                    "column": col_name,
                    "stat_test": col_stats.get("stat_test_name", "unknown"),
                    "drift_score": round(float(col_stats.get("drift_score", 0.0)), 4),
                }
            )

    summary = {
        "version_label": version_label,
        "dataset_drift_detected": bool(dataset_drift),
        "drift_share": round(float(drift_share), 4),
        "drift_threshold": drift_share_threshold,
        "number_of_drifted_features": int(number_of_drifted),
        "total_features_evaluated": int(total_columns),
        "drifted_features": drifted_features,
        "html_report": str(html_path),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DriftReport] Report generated: {html_path}")
    print(f"[DriftReport] Drift Detected: {dataset_drift} (Share: {drift_share:.2%} of features, Threshold: {drift_share_threshold:.2%})")
    print(f"[DriftReport] Drifted columns ({len(drifted_features)}): {[d['column'] for d in drifted_features]}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate Evidently drift report.")
    parser.add_argument("--version", type=str, default="v1", help="Dataset version (v1, v2, v3)")
    parser.add_argument("--config", type=str, default="configs/monitoring.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["monitoring"]

    ref_path = Path(cfg["reference_data_path"])
    # Current path defaults to the replay or version features
    curr_path = Path(f"data/processed/features_{args.version}.parquet")
    if not curr_path.exists():
        curr_path = Path("data/processed/features.parquet")

    generate_drift_report(
        reference_path=ref_path,
        current_path=curr_path,
        output_dir=Path(cfg["reports_dir"]),
        version_label=args.version,
        drift_share_threshold=cfg["thresholds"]["drift_share_threshold"],
    )


if __name__ == "__main__":
    main()
