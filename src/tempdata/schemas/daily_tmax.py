"""Daily maximum temperature schema.

This is the most important schema because it maps directly to market resolution logic.

Key rules (do not bend these later):
- date_local is station local time, not UTC
- tmax_f exists for market compatibility
- coverage_hours lets you detect bad days
- qc_flags bubbles up any hourly issues

This table becomes:
- your training label
- your backtest truth
- your trading ground truth
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

from tempdata.schemas.validate import (
    require_close,
    require_columns,
    require_date_no_time,
    require_int_range,
    require_nonnegative_int,
    require_no_nulls,
    require_range,
    require_timezone_utc,
    require_unique,
)


class DailyTmax(TypedDict):
    """Daily maximum temperature record.

    Represents the highest temperature on a calendar date at a station.
    Preserves local calendar correctness for market resolution.
    """

    date_local: pd.Timestamp  # Local calendar date (midnight in station timezone)
    station_id: str  # Station identifier (e.g., "KLGA")
    tmax_c: float  # Maximum temperature in Celsius
    tmax_f: float  # Maximum temperature in Fahrenheit (for market compatibility)
    coverage_hours: int  # Number of hourly observations used
    source: str  # Data source identifier (e.g., "noaa")
    qc_flags: int  # Quality control flags (bitmask, 0 = OK)
    updated_at_utc: pd.Timestamp  # When this record was last updated (UTC)


# Column order for DataFrame operations
DAILY_TMAX_FIELDS = [
    "date_local",
    "station_id",
    "tmax_c",
    "tmax_f",
    "coverage_hours",
    "source",
    "qc_flags",
    "updated_at_utc",
]

# Required columns for validation
REQUIRED_COLUMNS = DAILY_TMAX_FIELDS.copy()

# Dataset name for error messages
_DATASET_NAME = "daily_tmax"


def _celsius_to_fahrenheit(celsius: pd.Series) -> pd.Series:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def validate_daily_tmax(df: pd.DataFrame) -> None:
    """Validate that a DataFrame conforms to the daily_tmax schema.

    Checks performed:
    - All required columns present
    - Uniqueness on (date_local, station_id)
    - coverage_hours is int in [0, 24]
    - tmax_c in [-90, 60], tmax_f in [-130, 140]
    - updated_at_utc is tz-aware UTC
    - date_local has no time component (midnight only)
    - Fahrenheit/Celsius consistency: |tmax_f - (tmax_c * 9/5 + 32)| <= 0.2

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If any validation check fails
    """
    # Required columns
    require_columns(df.columns, REQUIRED_COLUMNS, dataset=_DATASET_NAME)

    # Empty DataFrame is valid (no rows to check)
    if df.empty:
        return

    # Uniqueness check
    require_unique(df, ["date_local", "station_id"], dataset=_DATASET_NAME)

    # Non-null columns
    require_no_nulls(
        df,
        ["date_local", "station_id", "tmax_c", "tmax_f", "coverage_hours", "source", "qc_flags", "updated_at_utc"],
        dataset=_DATASET_NAME,
    )

    # Range checks
    require_int_range(df, "coverage_hours", lo=0, hi=24, dataset=_DATASET_NAME)
    require_range(df, "tmax_c", lo=-90, hi=60, dataset=_DATASET_NAME)
    require_range(df, "tmax_f", lo=-130, hi=140, dataset=_DATASET_NAME)

    # qc_flags must be non-negative
    require_nonnegative_int(df, "qc_flags", dataset=_DATASET_NAME)

    # Timezone check for updated_at_utc
    require_timezone_utc(df, "updated_at_utc", dataset=_DATASET_NAME)

    # date_local should be at midnight (no time component)
    require_date_no_time(df, "date_local", dataset=_DATASET_NAME)

    # Fahrenheit/Celsius consistency check
    require_close(
        df,
        "tmax_c",
        "tmax_f",
        tol=0.2,
        transform_a_to_b=_celsius_to_fahrenheit,
        dataset=_DATASET_NAME,
    )
