"""Build daily maximum temperature aggregates.

This stage:
- Groups hourly observations by local calendar date
- Computes daily Tmax in both Celsius and Fahrenheit
- Tracks coverage (number of valid hourly observations per day)
- Bubbles up QC flags from hourly data using bitwise OR
- Excludes QC_OUT_OF_RANGE observations from Tmax calculation

The output is validated against the daily_tmax schema.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import reduce
from pathlib import Path

import pandas as pd

from tempdata.schemas.daily_tmax import DAILY_TMAX_FIELDS, validate_daily_tmax
from tempdata.schemas.qc_flags import (
    QC_INCOMPLETE_DAY,
    QC_LOW_COVERAGE,
    QC_OUT_OF_RANGE,
)


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def _bitwise_or_agg(series: pd.Series) -> int:
    """Aggregate QC flags using bitwise OR."""
    return reduce(lambda a, b: a | b, series, 0)


def build_daily_tmax(
    hourly_df: pd.DataFrame,
    station_tz: str,
    min_coverage_hours: int = 18,
) -> pd.DataFrame:
    """Aggregate hourly observations to daily Tmax.

    An hourly observation is eligible for Tmax calculation if:
    - temp_c is not null
    - QC_OUT_OF_RANGE is NOT set

    Spike-flagged values (QC_SPIKE_DETECTED) ARE included in Tmax calculation
    to avoid accidentally removing real heat spikes. The spike flag is
    propagated to the daily level for downstream awareness.

    Args:
        hourly_df: DataFrame with hourly_obs schema
        station_tz: Timezone string for the station (e.g., "America/New_York")
        min_coverage_hours: Minimum hours for a day to avoid QC_LOW_COVERAGE (default 18)

    Returns:
        DataFrame with daily_tmax schema
    """
    if hourly_df.empty:
        return pd.DataFrame(columns=DAILY_TMAX_FIELDS)

    df = hourly_df.copy()

    # Convert UTC to local time for date grouping
    df["ts_local"] = df["ts_utc"].dt.tz_convert(station_tz)
    df["date_local"] = df["ts_local"].dt.normalize()

    # Mark valid observations for Tmax calculation:
    # - temp_c is not null
    # - QC_OUT_OF_RANGE is NOT set
    # Note: spike-flagged values ARE included (per design doc)
    df["is_valid"] = df["temp_c"].notna() & ((df["qc_flags"] & QC_OUT_OF_RANGE) == 0)

    # For Tmax calculation, use only valid observations
    df["temp_c_valid"] = df["temp_c"].where(df["is_valid"])

    # Extract hour for coverage counting (NOAA ISD has sub-hourly data)
    # We count unique hours with valid observations, not total observations
    df["hour_local"] = df["ts_local"].dt.hour

    # For counting unique hours, we need the hour only where observation is valid
    df["hour_valid"] = df["hour_local"].where(df["is_valid"])

    # Group by local date and station
    grouped = df.groupby(["date_local", "station_id"])

    # Aggregate
    # coverage_hours = number of unique hours with valid observations (max 24)
    daily = grouped.agg(
        tmax_c=("temp_c_valid", "max"),
        coverage_hours=("hour_valid", "nunique"),
        qc_flags=("qc_flags", _bitwise_or_agg),
    ).reset_index()

    # Set source explicitly (per design doc: "noaa_isd")
    daily["source"] = "noaa_isd"

    # Add Fahrenheit
    daily["tmax_f"] = daily["tmax_c"].apply(celsius_to_fahrenheit).round(1)

    # Add timestamp
    daily["updated_at_utc"] = pd.Timestamp.now(tz=timezone.utc)

    # Flag incomplete days (coverage == 0)
    incomplete = daily["coverage_hours"] == 0
    daily.loc[incomplete, "qc_flags"] = daily.loc[incomplete, "qc_flags"] | QC_INCOMPLETE_DAY

    # Flag low coverage days (coverage < threshold, but > 0)
    low_coverage = (daily["coverage_hours"] < min_coverage_hours) & (daily["coverage_hours"] > 0)
    daily.loc[low_coverage, "qc_flags"] = daily.loc[low_coverage, "qc_flags"] | QC_LOW_COVERAGE

    # Remove days with no valid observations (tmax_c is null)
    # These days have coverage_hours == 0 and QC_INCOMPLETE_DAY flag
    daily = daily[daily["tmax_c"].notna()].reset_index(drop=True)

    # Ensure column order
    daily = daily[DAILY_TMAX_FIELDS]

    return daily


def write_daily_tmax(
    daily_df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Validate and write daily Tmax DataFrame to parquet.

    Args:
        daily_df: DataFrame with daily_tmax schema
        output_path: Path to write daily Tmax parquet

    Returns:
        Path to written output file

    Raises:
        ValueError: If output fails schema validation
    """
    output_path = Path(output_path)

    # Validate schema before writing
    validate_daily_tmax(daily_df)

    # Atomic write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    daily_df.to_parquet(tmp_path, index=False)
    tmp_path.rename(output_path)

    print(f"[aggregate] wrote {len(daily_df)} rows to {output_path}")
    return output_path


def aggregate_to_daily_tmax(
    input_path: Path | str,
    output_path: Path | str,
    station_tz: str,
    min_coverage_hours: int = 18,
) -> Path:
    """Read hourly observations, aggregate to daily Tmax, write output.

    Args:
        input_path: Path to cleaned hourly observations parquet
        output_path: Path to write daily Tmax parquet
        station_tz: Timezone string for the station
        min_coverage_hours: Minimum hours for a day to avoid QC_LOW_COVERAGE

    Returns:
        Path to written output file

    Raises:
        ValueError: If output fails schema validation
    """
    input_path = Path(input_path)

    hourly_df = pd.read_parquet(input_path)
    daily_df = build_daily_tmax(hourly_df, station_tz, min_coverage_hours)

    return write_daily_tmax(daily_df, output_path)
