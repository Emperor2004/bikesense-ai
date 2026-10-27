"""Module for generating US holiday indicator flags."""

from pathlib import Path

import holidays
import pandas as pd
import yaml


def build_holiday_flags(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate hourly holiday indicator flags for the date range.
    
    Args:
        start_date: String in YYYY-MM-DD format (inclusive start)
        end_date: String in YYYY-MM-DD format (inclusive end)
        
    Returns:
        DataFrame with columns ['datetime_hour', 'is_holiday']
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="h")
    years = list(range(pd.to_datetime(start_date).year, pd.to_datetime(end_date).year + 1))
    us_holidays = holidays.country_holidays("US", years=years)

    is_holiday = [1 if ts.date() in us_holidays else 0 for ts in date_range]

    return pd.DataFrame(
        {
            "datetime_hour": date_range,
            "is_holiday": is_holiday,
        }
    )


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    month_keys = sorted(params["months"].keys(), key=lambda k: int(k.replace("m", "")))
    first_month = params["months"][month_keys[0]]
    last_month = params["months"][month_keys[-1]]

    start_date = f"{first_month}-01 00:00:00"
    # End date: last day of last month at 23:00:00
    last_month_dt = pd.to_datetime(f"{last_month}-01")
    end_date = (last_month_dt + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d 23:00:00")

    print(f"Generating holidays from {start_date} to {end_date}...")
    holiday_df = build_holiday_flags(start_date, end_date)

    out_dir = Path("data/external")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "holidays.parquet"

    holiday_df.to_parquet(out_file, index=False)
    print(f"Saved {len(holiday_df)} holiday rows to {out_file}")


if __name__ == "__main__":
    main()
