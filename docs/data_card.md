# Data Card: BikeSense AI

## 1. Dataset Summary
- **Source 1: Citi Bike Trip Histories (NYC)**
  - Provider: NYCBS (Lyft / City of New York)
  - URL: `https://s3.amazonaws.com/tripdata`
  - License: NYCBS Data Use Policy (free for analysis and academic research)
  - Period: 14 consecutive months (July 2023 – August 2024)
- **Source 2: Open-Meteo Historical Weather API**
  - Provider: Open-Meteo.com (ERA5 Reanalysis)
  - License: CC BY 4.0
  - Variables: Hourly `temperature_2m` (°C), `precipitation` (mm), `wind_speed_10m` (km/h)
- **Source 3: US Federal Holidays**
  - Provider: Python `holidays` package

## 2. Dataset Versions & Splits
Fourteen months (M1–M14) are grouped into 3 cumulative versions:

| Version | Training Set | Simulated Production Replay |
|---|---|---|
| **v1** | M1–M8 (2023-07 to 2024-02) | M9–M10 (2024-03 to 2024-04) |
| **v2** | M1–M10 (2023-07 to 2024-04) | M11–M12 (2024-05 to 2024-06) |
| **v3** | M1–M12 (2023-07 to 2024-06) | M13–M14 (2024-07 to 2024-08) |

## 3. Station Selection
- Fixed to the **top 30 busiest departure stations** identified across the initial training window (Dataset v1).
- Canonical station key: `start_station_id`.

## 4. Feature Schema Contract
Final feature table format (`data/processed/features_{version}.parquet`):

| Column | Type | Description |
|---|---|---|
| `station_id` | string | Canonical Citi Bike station identifier |
| `datetime_hour` | timestamp | Timestamp rounded to start of hour |
| `departures` | integer | Target variable: departure trip count |
| `hour` | integer (0-23) | Hour of the day |
| `day_of_week` | integer (0-6) | Day of week (0=Monday, 6=Sunday) |
| `month` | integer (1-12) | Month of the year |
| `is_holiday` | integer (0/1) | Federal holiday indicator |
| `temperature_2m` | float | 2-meter air temperature (°C) |
| `precipitation` | float | Hourly precipitation (mm) |
| `wind_speed_10m` | float | Wind speed at 10 meters (km/h) |
| `lag_24h` | float | Departures at same station 24 hours prior |
| `lag_168h` | float | Departures at same station 168 hours (7 days) prior |
| `rolling_7d_same_hour_mean` | float | Mean departures across past 7 days for the same hour |

## 5. Leakage & Known Limitations
- **Issue Time:** Forecasts are produced at 00:00 for the next 24 hours. All lag and rolling calculations strictly look backward from 00:00.
- **Censored Demand:** Recorded trips capture fulfilled demand, not unmet demand when a station was empty.
- **Weather Reanalysis:** Historical weather uses ERA5 reanalysis rather than live meteorological forecast snapshots.
