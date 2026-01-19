"""Clean hourly observation data.

This stage:
- Validates input schema (early fail on malformed data)
- Deduplicates observations (by ts_utc, station_id)
- Flags missing temperature values
- Flags and nullifies out-of-range values
- Detects hour-to-hour spikes
- Validates output schema

Design principles:
- Cleaning != filtering: flag issues, don't delete aggressively
- Deterministic rules only: no ML, no future-dependent heuristics
- Idempotent: running twice should not change results
- Schema-safe: output must pass validate_hourly_obs
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tempdata.schemas.hourly_obs import RAW_HOURLY_FIELDS, validate_hourly_obs
from tempdata.schemas.qc_flags import (
    QC_DUPLICATE_TS,
    QC_MISSING_VALUE,
    QC_OUT_OF_RANGE,
    QC_SPIKE_DETECTED,
)
from tempdata.schemas.validate import (
    require_columns,
    require_no_nulls,
    require_nonnegative_int,
    require_range,
    require_timezone_utc,
)


def dedupe_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (ts_utc, station_id) rows, keeping first.

    Flags duplicates with QC_DUPLICATE_TS before removal.

    Args:
        df: DataFrame with ts_utc and station_id columns

    Returns:
        DataFrame with duplicates removed (first occurrence kept)
    """
    if df.empty:
        return df

    # Identify duplicates (all but first occurrence)
    dup_mask = df.duplicated(subset=["ts_utc", "station_id"], keep="first")

    # Flag duplicates before removing
    df = df.copy()
    df.loc[dup_mask, "qc_flags"] = df.loc[dup_mask, "qc_flags"] | QC_DUPLICATE_TS

    # Keep first occurrence only
    return df[~dup_mask].reset_index(drop=True)


def flag_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where temp_c is null.

    Does not remove rows - missing timestamps are preserved for coverage accounting.

    Args:
        df: DataFrame with temp_c column

    Returns:
        DataFrame with QC_MISSING_VALUE flag set on null temp_c rows
    """
    if df.empty:
        return df

    df = df.copy()
    missing = df["temp_c"].isna()
    df.loc[missing, "qc_flags"] = df.loc[missing, "qc_flags"] | QC_MISSING_VALUE
    return df


def flag_out_of_range(
    df: pd.DataFrame,
    temp_min: float = -90,
    temp_max: float = 60,
) -> pd.DataFrame:
    """Flag and nullify temperature values outside reasonable range.

    Uses very wide physical bounds (not climate bounds):
    - Valid range: [-90, 60] C

    Out-of-range values are:
    1. Flagged with QC_OUT_OF_RANGE
    2. Set to NaN (prevents invalid values from becoming Tmax)

    Args:
        df: DataFrame with temp_c column
        temp_min: Minimum valid temperature in Celsius (default -90)
        temp_max: Maximum valid temperature in Celsius (default 60)

    Returns:
        DataFrame with out-of-range values flagged and nullified
    """
    if df.empty:
        return df

    df = df.copy()
    out_of_range = (df["temp_c"] < temp_min) | (df["temp_c"] > temp_max)
    df.loc[out_of_range, "qc_flags"] = df.loc[out_of_range, "qc_flags"] | QC_OUT_OF_RANGE
    df.loc[out_of_range, "temp_c"] = None
    return df


def flag_spikes(df: pd.DataFrame, threshold: float = 15.0) -> pd.DataFrame:
    """Flag hour-to-hour temperature spikes.

    A spike is detected when |temp[t] - temp[t-1]| > threshold.
    This catches sensor glitches.

    Note:
    - Does NOT delete or auto-correct spikes
    - Let aggregation decide whether to exclude flagged hours
    - Assumes data is sorted by ts_utc

    Args:
        df: DataFrame with temp_c column (must be sorted by ts_utc)
        threshold: Maximum allowed temperature change in C (default 15)

    Returns:
        DataFrame with QC_SPIKE_DETECTED flag set on spike rows
    """
    if df.empty or len(df) < 2:
        return df

    df = df.copy()
    delta = df["temp_c"].diff().abs()
    spike = delta > threshold
    df.loc[spike, "qc_flags"] = df.loc[spike, "qc_flags"] | QC_SPIKE_DETECTED
    return df


def _validate_input_schema(df: pd.DataFrame) -> None:
    """Validate input schema for cleaning (lighter than full validation).

    Checks structure and types, but NOT temp_c range (we'll fix that during cleaning).
    This catches truly malformed data while allowing data quality issues through.

    Raises:
        ValueError: If structural validation fails
    """
    dataset = "hourly_obs_input"

    # Required columns
    require_columns(df.columns, RAW_HOURLY_FIELDS, dataset=dataset)

    if df.empty:
        return

    # Timezone check
    require_timezone_utc(df, "ts_utc", dataset=dataset)

    # Non-null columns (temp_c can be null for missing observations)
    require_no_nulls(df, ["ts_utc", "station_id", "source", "qc_flags"], dataset=dataset)

    # Coordinate range checks (these should never be wrong)
    require_range(df, "lat", lo=-90, hi=90, allow_null=True, dataset=dataset)
    require_range(df, "lon", lo=-180, hi=180, allow_null=True, dataset=dataset)

    # qc_flags must be non-negative
    require_nonnegative_int(df, "qc_flags", dataset=dataset)

    # NOTE: We intentionally skip temp_c range check here
    # Out-of-range temps are data quality issues, not schema issues
    # They will be flagged and fixed in the cleaning steps


def print_cleaning_stats(
    df: pd.DataFrame,
    original_count: int,
    duplicates_removed: int,
) -> None:
    """Print summary statistics after cleaning.

    Args:
        df: Cleaned DataFrame
        original_count: Number of rows before cleaning
        duplicates_removed: Number of duplicate rows removed
    """
    print(f"[clean] Cleaning summary:")
    print(f"  Total rows: {original_count} -> {len(df)} ({duplicates_removed} duplicates removed)")

    # Count rows with any QC flag
    flagged_count = (df["qc_flags"] != 0).sum()
    print(f"  Rows with QC flags: {flagged_count}")

    # Count by individual flag
    flag_names = {
        QC_MISSING_VALUE: "QC_MISSING_VALUE",
        QC_OUT_OF_RANGE: "QC_OUT_OF_RANGE",
        QC_SPIKE_DETECTED: "QC_SPIKE_DETECTED",
        QC_DUPLICATE_TS: "QC_DUPLICATE_TS",
    }

    for flag_val, flag_name in flag_names.items():
        count = ((df["qc_flags"] & flag_val) != 0).sum()
        if count > 0:
            print(f"    {flag_name}: {count}")

    # Min/max temp (non-null only)
    valid_temps = df["temp_c"].dropna()
    if len(valid_temps) > 0:
        print(f"  Temp range (valid): {valid_temps.min():.1f}C to {valid_temps.max():.1f}C")
    else:
        print("  Temp range: no valid temperatures")


def clean_hourly_obs(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Clean hourly observations DataFrame.

    Cleaning steps (in order):
    1. Validate input schema (early fail on malformed data)
    2. Sort by ts_utc
    3. Flag and remove duplicate (ts_utc, station_id) rows
    4. Flag missing temperature values
    5. Flag and nullify out-of-range temperatures
    6. Flag hour-to-hour spikes
    7. Validate output schema

    Design principles:
    - Flag, don't delete (except true duplicates)
    - Deterministic, idempotent rules
    - Preserves all timestamps for coverage accounting

    Args:
        df: Raw hourly observations DataFrame
        verbose: If True, print cleaning statistics (default True)

    Returns:
        Cleaned DataFrame with qc_flags updated

    Raises:
        ValueError: If input or output fails schema validation
    """
    # Step 1: Validate input schema (early fail)
    # We check structure and types, but NOT temp_c range (we'll fix that in step 5)
    _validate_input_schema(df)

    original_count = len(df)

    # Ensure column order
    df = df[RAW_HOURLY_FIELDS].copy()

    # Step 2: Sort by ts_utc
    df = df.sort_values("ts_utc").reset_index(drop=True)

    # Step 3: Flag and remove duplicates
    pre_dedupe_count = len(df)
    df = dedupe_hourly(df)
    duplicates_removed = pre_dedupe_count - len(df)

    # Step 4: Flag missing temperature values
    df = flag_missing_values(df)

    # Step 5: Flag and nullify out-of-range temperatures
    df = flag_out_of_range(df)

    # Step 6: Flag spikes
    df = flag_spikes(df)

    # Step 7: Validate output schema
    validate_hourly_obs(df, require_unique_keys=True)

    # Report statistics
    if verbose:
        print_cleaning_stats(df, original_count, duplicates_removed)

    return df


def clean_hourly_file(
    input_path: Path | str,
    output_path: Path | str,
    verbose: bool = True,
) -> Path:
    """Read, clean, and write hourly parquet file.

    This is a convenience wrapper around clean_hourly_obs for file-based workflows.

    Args:
        input_path: Path to raw hourly observations parquet
        output_path: Path to write cleaned observations
        verbose: If True, print cleaning statistics

    Returns:
        Path to written output file

    Raises:
        ValueError: If input or output fails schema validation
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_parquet(input_path)
    df = clean_hourly_obs(df, verbose=verbose)

    # Atomic write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.rename(output_path)

    if verbose:
        print(f"[clean] Wrote {len(df)} rows to {output_path}")

    return output_path
