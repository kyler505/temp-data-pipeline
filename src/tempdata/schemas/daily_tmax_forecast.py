"""Daily maximum temperature forecast schema.

This schema defines the structure for forecast data from Open-Meteo and other sources.
It enables joining forecasts to truth data (daily_tmax) for modeling and backtesting.

Key rules:
- issue_time_utc is when the forecast was fetched/issued (tz-aware UTC)
- target_date_local is the station-local calendar date being forecasted (midnight, tz-naive)
- lead_hours is computed from issue time to target date midnight in station tz
- Both Celsius and Fahrenheit are stored for consistency with daily_tmax
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

from tempdata.schemas.validate import (
    require_close,
    require_columns,
    require_date_no_time,
    require_int_range,
    require_no_nulls,
    require_range,
    require_timezone_utc,
    require_unique,
)


class DailyTmaxForecast(TypedDict):
    """Daily maximum temperature forecast record.

    Represents a forecast for the highest temperature on a calendar date at a station.
    """

    station_id: str  # Station identifier (e.g., "KLGA")
    lat: float  # Latitude of station
    lon: float  # Longitude of station
    issue_time_utc: pd.Timestamp  # When forecast was issued/fetched (tz-aware UTC)
    target_date_local: pd.Timestamp  # Local calendar date (midnight, tz-naive)
    tmax_pred_c: float  # Predicted maximum temperature in Celsius
    tmax_pred_f: float  # Predicted maximum temperature in Fahrenheit
    lead_hours: int  # Hours from issue time to target date midnight (in station tz)
    model: str  # Forecast model identifier (e.g., "openmeteo")
    source: str  # Data source identifier (e.g., "openmeteo")
    ingested_at_utc: pd.Timestamp  # When this record was ingested (tz-aware UTC)


# Column order for DataFrame operations
DAILY_TMAX_FORECAST_FIELDS = [
    "station_id",
    "lat",
    "lon",
    "issue_time_utc",
    "target_date_local",
    "tmax_pred_c",
    "tmax_pred_f",
    "lead_hours",
    "model",
    "source",
    "ingested_at_utc",
]

# Required columns for validation
REQUIRED_COLUMNS = DAILY_TMAX_FORECAST_FIELDS.copy()

# Dataset name for error messages
_DATASET_NAME = "daily_tmax_forecast"


def _celsius_to_fahrenheit(celsius: pd.Series) -> pd.Series:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def validate_daily_tmax_forecast(df: pd.DataFrame) -> None:
    """Validate that a DataFrame conforms to the daily_tmax_forecast schema.

    Checks performed:
    - All required columns present
    - Uniqueness on (station_id, issue_time_utc, target_date_local)
    - issue_time_utc and ingested_at_utc are tz-aware UTC
    - target_date_local is at midnight (no time component)
    - tmax_pred_c in [-90, 60], tmax_pred_f in [-130, 140]
    - lead_hours in [-24, 720] (30 days)
    - Fahrenheit/Celsius consistency: |tmax_pred_f - (tmax_pred_c * 9/5 + 32)| <= 0.2

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
    require_unique(
        df, ["station_id", "issue_time_utc", "target_date_local"], dataset=_DATASET_NAME
    )

    # Non-null columns
    require_no_nulls(df, REQUIRED_COLUMNS, dataset=_DATASET_NAME)

    # Timezone checks
    require_timezone_utc(df, "issue_time_utc", dataset=_DATASET_NAME)
    require_timezone_utc(df, "ingested_at_utc", dataset=_DATASET_NAME)

    # target_date_local should be at midnight (no time component)
    require_date_no_time(df, "target_date_local", dataset=_DATASET_NAME)

    # Range checks
    require_range(df, "tmax_pred_c", lo=-90, hi=60, dataset=_DATASET_NAME)
    require_range(df, "tmax_pred_f", lo=-130, hi=140, dataset=_DATASET_NAME)
    require_int_range(df, "lead_hours", lo=-24, hi=720, dataset=_DATASET_NAME)

    # Latitude/longitude sanity
    require_range(df, "lat", lo=-90, hi=90, dataset=_DATASET_NAME)
    require_range(df, "lon", lo=-180, hi=180, dataset=_DATASET_NAME)

    # Fahrenheit/Celsius consistency check
    require_close(
        df,
        "tmax_pred_c",
        "tmax_pred_f",
        tol=0.2,
        transform_a_to_b=_celsius_to_fahrenheit,
        dataset=_DATASET_NAME,
    )
