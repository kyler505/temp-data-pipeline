"""Canonical hourly observation schema.

This defines what every hourly observation must look like, regardless of source.
This schema is what:
- your fetcher outputs
- your cleaner expects
- your daily aggregation consumes

Non-negotiables:
- ts_utc is timezone-aware UTC
- temp_c is Celsius
- Fahrenheit never appears here
- qc_flags exists even if always 0 at fetch time
"""

from __future__ import annotations

from typing import Iterable, TypedDict

import pandas as pd

from tempdata.schemas.validate import (
    require_columns,
    require_nonnegative_int,
    require_no_nulls,
    require_range,
    require_timezone_utc,
    require_unique,
)


class HourlyObs(TypedDict):
    """Canonical hourly observation record.

    All hourly data in the pipeline must conform to this structure.
    """

    ts_utc: pd.Timestamp  # Timezone-aware UTC timestamp
    station_id: str  # Station identifier (e.g., "KLGA")
    lat: float  # Latitude in degrees
    lon: float  # Longitude in degrees
    temp_c: float  # Temperature in Celsius (may be NaN for missing)
    source: str  # Data source identifier (e.g., "noaa")
    qc_flags: int  # Quality control flags (bitmask, 0 = OK)


# Column order for DataFrame operations
RAW_HOURLY_FIELDS = [
    "ts_utc",
    "station_id",
    "lat",
    "lon",
    "temp_c",
    "source",
    "qc_flags",
]

# Required columns for validation
REQUIRED_COLUMNS = RAW_HOURLY_FIELDS.copy()

# Dataset name for error messages
_DATASET_NAME = "hourly_obs"


def ensure_hourly_schema_columns(columns: Iterable[str]) -> list[str]:
    """Return columns ordered to the raw hourly schema.

    Raises ValueError if required columns are missing.
    """
    col_set = set(columns)
    missing = [col for col in RAW_HOURLY_FIELDS if col not in col_set]
    if missing:
        raise ValueError(f"Missing raw hourly fields: {missing}")
    return list(RAW_HOURLY_FIELDS)


def validate_hourly_obs(
    df: pd.DataFrame,
    require_unique_keys: bool = True,
) -> None:
    """Validate that a DataFrame conforms to the hourly_obs schema.

    Checks performed:
    - All required columns present
    - ts_utc is tz-aware UTC
    - No nulls in: ts_utc, station_id, source, qc_flags
    - temp_c in [-90, 60] (wide range, allows nulls for missing data)
    - lat in [-90, 90], lon in [-180, 180]
    - qc_flags is non-negative integer
    - Uniqueness on (ts_utc, station_id) if require_unique_keys=True

    Args:
        df: DataFrame to validate
        require_unique_keys: If True, enforce uniqueness on (ts_utc, station_id).
            Set to False for fetch stage where deduplication happens in clean stage.

    Raises:
        ValueError: If any validation check fails
    """
    # Required columns
    require_columns(df.columns, REQUIRED_COLUMNS, dataset=_DATASET_NAME)

    # Empty DataFrame is valid (no rows to check)
    if df.empty:
        return

    # Timezone check
    require_timezone_utc(df, "ts_utc", dataset=_DATASET_NAME)

    # Non-null columns (temp_c can be null for missing observations)
    require_no_nulls(df, ["ts_utc", "station_id", "source", "qc_flags"], dataset=_DATASET_NAME)

    # Range checks
    require_range(df, "temp_c", lo=-90, hi=60, allow_null=True, dataset=_DATASET_NAME)
    require_range(df, "lat", lo=-90, hi=90, allow_null=True, dataset=_DATASET_NAME)
    require_range(df, "lon", lo=-180, hi=180, allow_null=True, dataset=_DATASET_NAME)

    # qc_flags must be non-negative
    require_nonnegative_int(df, "qc_flags", dataset=_DATASET_NAME)

    # Uniqueness check (optional - may skip during fetch, enforce after clean)
    if require_unique_keys:
        require_unique(df, ["ts_utc", "station_id"], dataset=_DATASET_NAME)
