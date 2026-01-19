"""Clean hourly observation data.

This stage:
- Deduplicates observations (by ts_utc, station_id)
- Applies QC flags for out-of-range values
- Normalizes data formats

The output is validated against the hourly_obs schema with uniqueness enforced.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tempdata.schemas.hourly_obs import RAW_HOURLY_FIELDS, validate_hourly_obs
from tempdata.schemas.qc_flags import QC_DUPLICATE_TS, QC_OUT_OF_RANGE


def dedupe_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (ts_utc, station_id) rows, keeping first.

    Flags duplicates in qc_flags before removal.
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


def flag_out_of_range(
    df: pd.DataFrame,
    temp_min: float = -90,
    temp_max: float = 60,
) -> pd.DataFrame:
    """Flag temperature values outside reasonable range.

    Does not remove rows, only sets QC_OUT_OF_RANGE flag.
    """
    if df.empty:
        return df

    df = df.copy()
    out_of_range = (df["temp_c"] < temp_min) | (df["temp_c"] > temp_max)
    df.loc[out_of_range, "qc_flags"] = df.loc[out_of_range, "qc_flags"] | QC_OUT_OF_RANGE
    return df


def clean_hourly_obs(
    input_path: Path | str,
    output_path: Path | str,
) -> Path:
    """Clean hourly observations from input parquet, write to output.

    Args:
        input_path: Path to raw hourly observations parquet
        output_path: Path to write cleaned observations

    Returns:
        Path to written output file

    Raises:
        ValueError: If output fails schema validation
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_parquet(input_path)

    # Ensure column order
    df = df[RAW_HOURLY_FIELDS]

    # Apply cleaning steps
    df = flag_out_of_range(df)
    df = dedupe_hourly(df)

    # Sort by timestamp
    df = df.sort_values("ts_utc").reset_index(drop=True)

    # Validate schema (with uniqueness enforced after deduplication)
    validate_hourly_obs(df, require_unique_keys=True)

    # Atomic write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.rename(output_path)

    print(f"[clean] wrote {len(df)} rows to {output_path}")
    return output_path
