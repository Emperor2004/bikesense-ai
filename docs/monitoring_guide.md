# Operational Monitoring Guide: BikeSense AI

## 1. Overview
BikeSense AI continuously audits model health in production across four complementary dimensions:
1. **Input Feature Drift:** Distribution shifts in meteorological and temporal lag signals.
2. **Prediction Drift:** Output distribution shifts in forecasted departure volumes.
3. **Model Performance Degradation:** Rolling 7-day MAE vs. the Champion model's reference training MAE.
4. **Operational Starvation Risk:** Under-forecast shortfall units representing unserved ride demand.

---

## 2. Threshold Calibration

| Metric | Threshold | Justification |
|---|---|---|
| **Drift Share** | `30%` (0.30) | Weather varies normally across seasons; alerting on individual features causes noise. Alerting when $>30\%$ of features drift simultaneously signals a systemic environmental shift. |
| **Statistical Test** | Wasserstein / KS ($p < 0.05$) | Non-parametric tests suitable for skewed departure counts and continuous weather features. |
| **MAE Degradation** | `1.25x` Reference MAE | A 25% increase in error beyond the champion's validation performance signals that the model has degraded beyond acceptable rebalancing tolerances. |
| **Shortfall Surge** | `1.20x` Baseline Rate | Captures asymmetric operational risk of empty bike stations. |

---

## 3. Alert Triage Runbook

### `ALERT-101`: Input Data Drift (Severity: WARNING)
- **Signal:** Share of drifted features $> 30\%$.
- **Interpretation:** Weather patterns or rider behavior have shifted (e.g. winter to spring transition).
- **Action:**
  1. Open `reports/monitoring/drift_report_{version}.html` to inspect which features drifted.
  2. If weather drifted (temperature, precipitation), verify whether model accuracy remains stable.
  3. If performance is still acceptable, no emergency action is needed; prepare the next dataset version for scheduled retraining.

### `ALERT-102`: Model Performance Degradation (Severity: CRITICAL)
- **Signal:** Rolling 7-day MAE $> 1.25\times$ Reference MAE.
- **Interpretation:** Model accuracy has significantly degraded on recent production actuals.
- **Action:**
  1. Inspect `reports/monitoring/rolling_error_series.csv`.
  2. Trigger the automated retraining workflow:
     ```bash
     gh workflow run retrain.yml -f dataset_version=v2
     ```
  3. Review the challenger's validation metrics via the promotion gate:
     ```bash
     python -m src.registry.promote --version v2
     ```

### `ALERT-103`: Shortfall Surge (Severity: HIGH)
- **Signal:** Unpredicted demand units increase $> 20\%$.
- **Interpretation:** Stations are in danger of running empty unexpectedly during commute peak hours.
- **Action:**
  1. Notify bike rebalancing operators to augment buffer stock at top stations.
  2. Check if a major holiday or city event occurred that was missing from calendar flags.

---

## 4. One-Step Rollback Runbook
If an newly promoted model behaves unexpectedly in production:
```bash
# Revert champion alias in <5 seconds without container restart or code change
python -m src.registry.rollback
```
The FastAPI service will automatically hot-reload the previous champion on the very next prediction request.
