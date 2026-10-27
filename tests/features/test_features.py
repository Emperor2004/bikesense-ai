"""Unit tests for feature engineering and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from src.data.holidays import build_holiday_flags
from src.features.build_features import (
    add_lag_features,
    add_rolling_features,
    add_temporal_features,
)


@pytest.fixture
def sample_hourly_departures():
    # 10 days of hourly continuous data for 2 stations = 240 hours
    dates = pd.date_range("2023-07-01 00:00:00", periods=240, freq="h")
    records = []
    for st in ["st_A", "st_B"]:
        for i, dt in enumerate(dates):
            # assign a known deterministic departure value: e.g. hour + station offset
            offset = 100 if st == "st_B" else 0
            records.append(
                {
                    "station_id": st,
                    "datetime_hour": dt,
                    "departures": dt.hour + offset,
                }
            )
    return pd.DataFrame(records)


def test_temporal_features(sample_hourly_departures):
    df = add_temporal_features(sample_hourly_departures)
    assert "hour" in df.columns
    assert "day_of_week" in df.columns
    assert "month" in df.columns

    assert df["hour"].min() >= 0 and df["hour"].max() <= 23
    assert df["day_of_week"].min() >= 0 and df["day_of_week"].max() <= 6
    assert (df["month"] == 7).all()


def test_no_leakage_lag_24h(sample_hourly_departures):
    df = add_temporal_features(sample_hourly_departures)
    df_lag = add_lag_features(df)

    # Check for st_A at index 24 (2023-07-02 00:00:00)
    st_a = df_lag[df_lag["station_id"] == "st_A"].sort_values("datetime_hour").reset_index(drop=True)

    # Lag 24 at index 24 must match departures at index 0
    assert np.isnan(st_a.loc[0, "lag_24h"])
    assert st_a.loc[24, "lag_24h"] == st_a.loc[0, "departures"]
    # Verify at hour 14 on day 2
    assert st_a.loc[38, "lag_24h"] == st_a.loc[14, "departures"]


def test_no_leakage_lag_168h(sample_hourly_departures):
    df = add_temporal_features(sample_hourly_departures)
    df_lag = add_lag_features(df)

    st_a = df_lag[df_lag["station_id"] == "st_A"].sort_values("datetime_hour").reset_index(drop=True)

    # 168 hours = exactly 7 days
    assert np.isnan(st_a.loc[167, "lag_168h"])
    assert st_a.loc[168, "lag_168h"] == st_a.loc[0, "departures"]
    assert st_a.loc[175, "lag_168h"] == st_a.loc[7, "departures"]


def test_no_leakage_rolling_mean(sample_hourly_departures):
    df = add_temporal_features(sample_hourly_departures)
    df_lags = add_lag_features(df)
    df_rolling = add_rolling_features(df_lags)

    st_a = df_rolling[df_rolling["station_id"] == "st_A"]
    # Check 8 AM across days
    st_a_8am = st_a[st_a["hour"] == 8].sort_values("datetime_hour").reset_index(drop=True)

    # In sample_hourly_departures, departures at 8 AM is always 8
    # Days 0 to 6 (7 days) should mean the rolling value on Day 7 (index 7) exists
    assert np.isnan(st_a_8am.loc[6, "rolling_7d_same_hour_mean"])
    assert st_a_8am.loc[7, "rolling_7d_same_hour_mean"] == 8.0

    # Strict leakage check: If we artificially change the 8th day's actual departures,
    # its rolling_7d_same_hour_mean MUST NOT change because it strictly looks at past 7 days!
    modified = sample_hourly_departures.copy()
    # Change departures on Day 7 at 8 AM to 999
    day7_8am_mask = (
        (modified["station_id"] == "st_A")
        & (modified["datetime_hour"] == pd.to_datetime("2023-07-08 08:00:00"))
    )
    modified.loc[day7_8am_mask, "departures"] = 999

    mod_temp = add_temporal_features(modified)
    mod_lags = add_lag_features(mod_temp)
    mod_rolling = add_rolling_features(mod_lags)

    mod_st_a_8am = mod_rolling[
        (mod_rolling["station_id"] == "st_A") & (mod_rolling["hour"] == 8)
    ].sort_values("datetime_hour").reset_index(drop=True)

    # Day 7 (index 7) rolling mean must still be 8.0 (excluding current day 999)
    assert mod_st_a_8am.loc[7, "rolling_7d_same_hour_mean"] == 8.0


def test_holiday_flag_known_date():
    # 4th of July 2023 is Independence Day (US federal holiday)
    holidays_df = build_holiday_flags("2023-07-03 00:00:00", "2023-07-05 23:00:00")
    july_3rd = holidays_df[holidays_df["datetime_hour"].dt.date == pd.to_datetime("2023-07-03").date()]
    july_4th = holidays_df[holidays_df["datetime_hour"].dt.date == pd.to_datetime("2023-07-04").date()]

    assert (july_3rd["is_holiday"] == 0).all()
    assert (july_4th["is_holiday"] == 1).all()
