"""Build daily maximum temperature aggregates.

This stage:
- Groups hourly observations by local calendar date
- Computes daily Tmax in both Celsius and Fahrenheit
- Tracks coverage (number of hourly observations per day)
- Bubbles up QC flags from hourly data

The output is validated against the daily_tmax schema.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tempdata.schemas.daily_tmax import DAILY_TMAX_FIELDS, validate_daily_tmax
from tempdata.schemas.qc_flags import QC_LOW_COVERAGE


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def build_daily_tmax(
    hourly_df: pd.DataFrame,
    station_tz: str,
    min_coverage_hours: int = 20,
) -> pd.DataFrame:
    """Aggregate hourly observations to daily Tmax.

    Args:
        hourly_df: DataFrame with hourly_obs schema
        station_tz: Timezone string for the station (e.g., "America/New_York")
        min_coverage_hours: Minimum hours required for a "good" day

    Returns:
        DataFrame with daily_tmax schema
    """
    if hourly_df.empty:
        return pd.DataFrame(columns=DAILY_TMAX_FIELDS)

    df = hourly_df.copy()

    # Convert UTC to local time for date grouping
    df["ts_local"] = df["ts_utc"].dt.tz_convert(station_tz)
    df["date_local"] = df["ts_local"].dt.normalize()

    # Group by local date and station
    grouped = df.groupby(["date_local", "station_id"])

    # Aggregate
    daily = grouped.agg(
        tmax_c=("temp_c", "max"),
        coverage_hours=("temp_c", "count"),
        source=("source", "first"),
        qc_flags=("qc_flags", lambda x: x.max()),  # Bubble up worst flag
    ).reset_index()

    # Add Fahrenheit
    daily["tmax_f"] = daily["tmax_c"].apply(celsius_to_fahrenheit).round(1)

    # Add timestamp
    daily["updated_at_utc"] = pd.Timestamp.now(tz=timezone.utc)

    # Flag low coverage days
    low_coverage = daily["coverage_hours"] < min_coverage_hours
    daily.loc[low_coverage, "qc_flags"] = daily.loc[low_coverage, "qc_flags"] | QC_LOW_COVERAGE

    # Ensure column order
    daily = daily[DAILY_TMAX_FIELDS]

    return daily


def aggregate_to_daily_tmax(
    input_path: Path | str,
    output_path: Path | str,
    station_tz: str,
    min_coverage_hours: int = 20,
) -> Path:
    """Read hourly observations, aggregate to daily Tmax, write output.

    Args:
        input_path: Path to cleaned hourly observations parquet
        output_path: Path to write daily Tmax parquet
        station_tz: Timezone string for the station
        min_coverage_hours: Minimum hours required for a "good" day

    Returns:
        Path to written output file

    Raises:
        ValueError: If output fails schema validation
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    hourly_df = pd.read_parquet(input_path)
    daily_df = build_daily_tmax(hourly_df, station_tz, min_coverage_hours)

    # Validate schema before writing
    validate_daily_tmax(daily_df)

    # Atomic write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    daily_df.to_parquet(tmp_path, index=False)
    tmp_path.rename(output_path)

    print(f"[aggregate] wrote {len(daily_df)} rows to {output_path}")
    return output_path
