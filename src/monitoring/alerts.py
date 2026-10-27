"""Automated multi-signal alert engine for model degradation and input drift."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.monitoring.rolling_error import compute_rolling_performance


def evaluate_alerts(
    config_path: Path = Path("configs/monitoring.yaml"),
    version_label: str = "v1",
) -> list[dict[str, Any]]:
    """Evaluate monitoring rules and fire alerts if thresholds are breached."""
    with open(config_path, "r", encoding="utf-8") as f:
        mon_cfg = yaml.safe_load(f)["monitoring"]

    thresholds = mon_cfg["thresholds"]
    reports_dir = Path(mon_cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    active_alerts: list[dict[str, Any]] = []
    timestamp = datetime.now(UTC).isoformat()

    # 1. Rule ALERT-101: Input Feature Drift
    drift_json_path = reports_dir / f"drift_summary_{version_label}.json"
    if drift_json_path.exists():
        with open(drift_json_path, "r", encoding="utf-8") as f:
            drift_summary = json.load(f)

        drift_share = float(drift_summary.get("drift_share", 0.0))
        drift_threshold = float(thresholds["drift_share_threshold"])

        if drift_share > drift_threshold:
            drifted_names = [d["column"] for d in drift_summary.get("drifted_features", [])]
            active_alerts.append(
                {
                    "alert_id": "ALERT-101",
                    "severity": "WARNING",
                    "name": "Input Data Drift",
                    "timestamp": timestamp,
                    "condition": f"Drifted feature share ({drift_share:.1%}) exceeded threshold ({drift_threshold:.1%})",
                    "details": {
                        "drift_share": drift_share,
                        "drifted_features": drifted_names,
                    },
                    "recommended_action": "Inspect feature distributions; prepare retrain dataset release.",
                }
            )

    # 2. Rule ALERT-102: Performance Degradation (Rolling MAE > 1.25x Baseline)
    actuals_path = Path(mon_cfg["joined_actuals_path"])
    if actuals_path.exists():
        try:
            perf_summary = compute_rolling_performance(
                actuals_path=actuals_path,
                window_days=mon_cfg.get("rolling_window_days", 7),
                output_dir=reports_dir,
            )
            deg_factor = float(perf_summary["degradation_factor"])
            max_deg_factor = float(thresholds["mae_degradation_factor"])

            if deg_factor > max_deg_factor:
                active_alerts.append(
                    {
                        "alert_id": "ALERT-102",
                        "severity": "CRITICAL",
                        "name": "Model Performance Degradation",
                        "timestamp": timestamp,
                        "condition": f"Rolling 7-day MAE ({perf_summary['latest_rolling_7d_mae']}) is {deg_factor}x reference baseline ({perf_summary['reference_mae']}), exceeding limit ({max_deg_factor}x)",
                        "details": perf_summary,
                        "recommended_action": "Trigger automated retraining workflow on newly arrived dataset window.",
                    }
                )

            # 3. Rule ALERT-103: Shortfall Surge
            # If rolling shortfall is noticeably high relative to daily MAE
            rolling_shortfall = float(perf_summary["latest_rolling_7d_shortfall"])
            daily_mae = float(perf_summary["latest_daily_mae"])
            if rolling_shortfall > (daily_mae * 1000 * float(thresholds.get("shortfall_surge_factor", 1.20))):
                active_alerts.append(
                    {
                        "alert_id": "ALERT-103",
                        "severity": "HIGH",
                        "name": "Under-forecast Shortfall Surge",
                        "timestamp": timestamp,
                        "condition": f"Elevated under-forecast shortfall detected ({rolling_shortfall:.1f} units)",
                        "details": {"rolling_7d_shortfall": rolling_shortfall},
                        "recommended_action": "Notify bike rebalancing operations to prevent empty station starvation.",
                    }
                )
        except Exception as e:
            print(f"Warning: Could not evaluate rolling performance: {e}")

    # Output active alerts report
    alerts_file = reports_dir / "active_alerts.json"
    with open(alerts_file, "w", encoding="utf-8") as f:
        json.dump(active_alerts, f, indent=2)

    print("\n=======================================================")
    print("             BikeSense AI Alert Assessment             ")
    print("=======================================================")
    if not active_alerts:
        print("[STATUS: OK] All monitored systems within normal operating bounds.")
    else:
        print(f"[STATUS: ALERTS ACTIVE] Found {len(active_alerts)} active alert(s):\n")
        for a in active_alerts:
            print(f"[{a['severity']}] {a['alert_id']}: {a['name']}")
            print(f"  Condition: {a['condition']}")
            print(f"  Action:    {a['recommended_action']}\n")

    print(f"Saved active alerts payload to: {alerts_file}")
    return active_alerts


def main():
    parser = argparse.ArgumentParser(description="Evaluate monitoring alert rules.")
    parser.add_argument("--config", type=str, default="configs/monitoring.yaml")
    parser.add_argument("--version", type=str, default="v1")
    args = parser.parse_args()

    alerts = evaluate_alerts(config_path=Path(args.config), version_label=args.version)
    has_critical = any(a["severity"] == "CRITICAL" for a in alerts)
    # Exit with code 2 if a critical degradation alert fired
    sys.exit(2 if has_critical else 0)


if __name__ == "__main__":
    main()
