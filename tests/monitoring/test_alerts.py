"""Unit tests for monitoring alert rules and threshold triggers."""

import json

import yaml

from src.monitoring.alerts import evaluate_alerts


def test_input_drift_alert_trigger(tmp_path):
    config_file = tmp_path / "monitoring.yaml"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "monitoring": {
            "reports_dir": str(reports_dir),
            "joined_actuals_path": str(tmp_path / "non_existent.parquet"),
            "thresholds": {
                "drift_share_threshold": 0.30,
                "mae_degradation_factor": 1.25,
            },
        }
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    # Mock drift summary JSON with 50% drift (above 30% threshold)
    drift_summary = {
        "drift_share": 0.50,
        "dataset_drift_detected": True,
        "drifted_features": [{"column": "temperature_2m"}],
    }
    with open(reports_dir / "drift_summary_v1.json", "w", encoding="utf-8") as f:
        json.dump(drift_summary, f)

    alerts = evaluate_alerts(config_path=config_file, version_label="v1")
    assert len(alerts) == 1
    assert alerts[0]["alert_id"] == "ALERT-101"
    assert alerts[0]["severity"] == "WARNING"


def test_no_alert_when_within_bounds(tmp_path):
    config_file = tmp_path / "monitoring.yaml"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "monitoring": {
            "reports_dir": str(reports_dir),
            "joined_actuals_path": str(tmp_path / "non_existent.parquet"),
            "thresholds": {
                "drift_share_threshold": 0.30,
                "mae_degradation_factor": 1.25,
            },
        }
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    # Mock drift summary with 10% drift (below 30% threshold)
    drift_summary = {
        "drift_share": 0.10,
        "dataset_drift_detected": False,
        "drifted_features": [],
    }
    with open(reports_dir / "drift_summary_v1.json", "w", encoding="utf-8") as f:
        json.dump(drift_summary, f)

    alerts = evaluate_alerts(config_path=config_file, version_label="v1")
    assert len(alerts) == 0
