"""Unit tests for training split integrity and lineage helpers."""

import pandas as pd
import pytest

from src.models.train import (
    get_dvc_dataset_hash,
    get_git_commit,
    time_ordered_train_val_split,
)


def test_time_ordered_split_no_shuffle():
    # 3 consecutive months
    dates = pd.date_range("2023-07-01", "2023-09-30 23:00:00", freq="D")
    df = pd.DataFrame(
        {
            "datetime_hour": dates,
            "departures": range(len(dates)),
        }
    )

    train_months = ["2023-07", "2023-08", "2023-09"]
    train_df, val_df = time_ordered_train_val_split(df, train_months)

    # Validation month must be the last month (September)
    assert (val_df["datetime_hour"].dt.strftime("%Y-%m") == "2023-09").all()

    # Train months must only be July & August
    assert set(train_df["datetime_hour"].dt.strftime("%Y-%m").unique()) == {"2023-07", "2023-08"}

    # Strict time order check: max train < min val
    assert train_df["datetime_hour"].max() < val_df["datetime_hour"].min()


def test_time_ordered_split_leakage_detection():
    # Artificially create overlapping or out-of-order records
    df = pd.DataFrame(
        {
            "datetime_hour": [
                pd.to_datetime("2023-09-01"),  # Sept
                pd.to_datetime("2023-07-01"),  # July
            ],
            "departures": [1, 2],
        }
    )
    # If train months are out of chronological order
    train_months = ["2023-09", "2023-07"]
    with pytest.raises(ValueError, match="Data leakage detected"):
        time_ordered_train_val_split(df, train_months)


def test_git_commit_helper():
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_dvc_hash_helper():
    dvc_hash = get_dvc_dataset_hash()
    assert isinstance(dvc_hash, str)
    assert len(dvc_hash) > 0
