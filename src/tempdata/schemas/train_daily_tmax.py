"""Training dataset schema for daily maximum temperature prediction.

This schema defines the feature-engineered dataset used for model training.
It joins forecasts to observed truth and adds derived features.

Key rules:
- All rolling features are computed with .shift(1) to prevent lookahead
- Rows within warm-up period may have NaNs in rolling features
- tmax_actual_f is the label (from validated daily_tmax truth)
- No duplicates on (station_id, issue_time_utc, target_date_local)
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

from tempdata.schemas.validate import (
    require_columns,
    require_date_no_time,
    require_int_range,
    require_no_nulls,
    require_range,
    require_timezone_utc,
    require_unique,
)


class TrainDailyTmax(TypedDict):
    """Training record for daily maximum temperature prediction.

    Contains forecast features, seasonal encodings, rolling bias/error
    statistics, and the observed truth label.
    """

    # Metadata
    station_id: str  # Station identifier (e.g., "KLGA")
    issue_time_utc: pd.Timestamp  # When forecast was issued (tz-aware UTC)
    target_date_local: pd.Timestamp  # Local calendar date (midnight, tz-naive)

    # Core forecast features
    tmax_pred_f: float  # Predicted maximum temperature (°F)
    lead_hours: int  # Hours from issue time to target date
    forecast_source: str  # Forecast provider identifier (e.g., "openmeteo")

    # Seasonal & calendar features
    sin_doy: float  # Sine of day-of-year (seasonal encoding)
    cos_doy: float  # Cosine of day-of-year (seasonal encoding)
    month: int  # Month of target date (1-12)

    # Rolling bias features (backward-looking only)
    bias_7d: float  # 7-day rolling mean forecast error
    bias_14d: float  # 14-day rolling mean forecast error
    bias_30d: float  # 30-day rolling mean forecast error

    # Rolling error features (backward-looking only)
    rmse_14d: float  # 14-day rolling RMSE
    rmse_30d: float  # 30-day rolling RMSE

    # Lead-time uncertainty
    sigma_lead: float  # Historical residual std dev for this lead_hours

    # Label
    tmax_actual_f: float  # Observed maximum temperature (°F)


# Column order for DataFrame operations
TRAIN_DAILY_TMAX_FIELDS = [
    # Metadata
    "station_id",
    "issue_time_utc",
    "target_date_local",
    # Core forecast features
    "tmax_pred_f",
    "lead_hours",
    "forecast_source",
    # Seasonal features
    "sin_doy",
    "cos_doy",
    "month",
    # Rolling bias features
    "bias_7d",
    "bias_14d",
    "bias_30d",
    # Rolling error features
    "rmse_14d",
    "rmse_30d",
    # Lead-time uncertainty
    "sigma_lead",
    # Label
    "tmax_actual_f",
]

# Required columns for validation
REQUIRED_COLUMNS = TRAIN_DAILY_TMAX_FIELDS.copy()

# Columns that must not be null (after warm-up period exclusion)
NON_NULL_COLUMNS = [
    "station_id",
    "issue_time_utc",
    "target_date_local",
    "tmax_pred_f",
    "lead_hours",
    "forecast_source",
    "sin_doy",
    "cos_doy",
    "month",
    "tmax_actual_f",
]

# Rolling columns that may be null during warm-up period
ROLLING_COLUMNS = [
    "bias_7d",
    "bias_14d",
    "bias_30d",
    "rmse_14d",
    "rmse_30d",
    "sigma_lead",
]

# Dataset name for error messages
_DATASET_NAME = "train_daily_tmax"


def validate_train_daily_tmax(
    df: pd.DataFrame,
    allow_warmup_nulls: bool = True,
) -> None:
    """Validate that a DataFrame conforms to the train_daily_tmax schema.

    Checks performed:
    - All required columns present
    - Uniqueness on (station_id, issue_time_utc, target_date_local)
    - issue_time_utc is tz-aware UTC
    - target_date_local is at midnight (no time component)
    - tmax_pred_f and tmax_actual_f in [-130, 140] (°F)
    - lead_hours in [-24, 720]
    - month in [1, 12]
    - sin_doy and cos_doy in [-1, 1]
    - Core columns have no nulls

    Args:
        df: DataFrame to validate
        allow_warmup_nulls: If True, rolling columns may have NaN values
            for rows in the warm-up period (default True)

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
        df,
        ["station_id", "issue_time_utc", "target_date_local"],
        dataset=_DATASET_NAME,
    )

    # Non-null columns (core features that must always be present)
    require_no_nulls(df, NON_NULL_COLUMNS, dataset=_DATASET_NAME)

    # Rolling columns null check (optional)
    if not allow_warmup_nulls:
        require_no_nulls(df, ROLLING_COLUMNS, dataset=_DATASET_NAME)

    # Timezone check for issue_time_utc
    require_timezone_utc(df, "issue_time_utc", dataset=_DATASET_NAME)

    # target_date_local should be at midnight (no time component)
    require_date_no_time(df, "target_date_local", dataset=_DATASET_NAME)

    # Range checks for temperature values
    require_range(df, "tmax_pred_f", lo=-130, hi=140, dataset=_DATASET_NAME)
    require_range(df, "tmax_actual_f", lo=-130, hi=140, dataset=_DATASET_NAME)

    # Range check for lead_hours
    require_int_range(df, "lead_hours", lo=-24, hi=720, dataset=_DATASET_NAME)

    # Range check for month
    require_int_range(df, "month", lo=1, hi=12, dataset=_DATASET_NAME)

    # Range checks for seasonal encodings
    require_range(df, "sin_doy", lo=-1, hi=1, dataset=_DATASET_NAME)
    require_range(df, "cos_doy", lo=-1, hi=1, dataset=_DATASET_NAME)

    # Range checks for rolling features (allow nulls for warm-up)
    require_range(
        df, "bias_7d", lo=-50, hi=50, allow_null=True, dataset=_DATASET_NAME
    )
    require_range(
        df, "bias_14d", lo=-50, hi=50, allow_null=True, dataset=_DATASET_NAME
    )
    require_range(
        df, "bias_30d", lo=-50, hi=50, allow_null=True, dataset=_DATASET_NAME
    )
    require_range(
        df, "rmse_14d", lo=0, hi=50, allow_null=True, dataset=_DATASET_NAME
    )
    require_range(
        df, "rmse_30d", lo=0, hi=50, allow_null=True, dataset=_DATASET_NAME
    )
    require_range(
        df, "sigma_lead", lo=0, hi=50, allow_null=True, dataset=_DATASET_NAME
    )
