# Model Card: BikeSense Demand Forecaster

## Model Details
- **Model Name:** `bikesense-demand-forecaster`
- **Model Family:** LightGBM Regressor (`LGBMRegressor`)
- **Task:** 24-hour ahead station-level hourly departure demand forecasting.
- **Serving Input:** 10 engineered features:
  - Temporal: `hour` (0–23), `day_of_week` (0–6), `month` (1–12), `is_holiday` (0/1).
  - Weather: `temperature_2m` (°C), `precipitation` (mm), `wind_speed_10m` (km/h).
  - Lags: `lag_24h` (departures 24h ago), `lag_168h` (departures 7d ago).
  - Rolling: `rolling_7d_same_hour_mean` (past 7 days demand at the same hour, excluding current day).
- **Target:** `departures` (integer >= 0).

## Training Protocol
- **Dataset Versions:** 
  - `v1`: Train M1–M7 (Jul 2023–Jan 2024), Val M8 (Feb 2024).
  - `v2`: Train M1–M9 (Jul 2023–Mar 2024), Val M10 (Apr 2024).
  - `v3`: Train M1–M11 (Jul 2023–May 2024), Val M12 (Jun 2024).
- **Split Type:** Strict chronological time-ordered splits (zero shuffling, zero future leakage).

## Evaluation Metrics
1. **MAE (Primary):** Mean absolute error in departures per station per hour.
2. **RMSE:** Root mean squared error penalizing large surge under/over-predictions.
3. **Under-forecast Shortfall:** $\sum \max(0, y - \hat{y})$ measuring total unpredicted demand units (operational starvation risk).

## Known Limitations
- **Censored Demand:** Recorded trips capture only fulfilled rentals, understating true demand when a station was empty.
- **Weather Reanalysis:** Historical weather uses ERA5 reanalysis rather than real-time numerical forecasts.
- **Station Scope:** Fixed to top 30 stations established from Dataset v1.
