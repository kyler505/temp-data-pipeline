"""Build training dataset for daily maximum temperature prediction.

This module transforms daily Tmax forecasts into a model-ready feature set:
1. Joins forecasts to observed truth (daily_tmax)
2. Adds seasonal & calendar encodings
3. Computes rolling bias/error statistics
4. Validates output against train_daily_tmax schema

All features are causality-safe: they only use information available
at forecast issue time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tempdata.features.rolling_stats import compute_all_rolling_features
from tempdata.schemas.train_daily_tmax import (
    TRAIN_DAILY_TMAX_FIELDS,
    validate_train_daily_tmax,
)


def join_forecast_to_truth(
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    min_coverage_hours: int = 18,
) -> pd.DataFrame:
    """Join forecasts to observed truth, filtering low-quality days.

    Performs an inner join on (station_id, target_date_local == date_local).
    Rows with low coverage in the truth data are excluded to ensure
    reliable labels.

    Args:
        forecast_df: DataFrame with daily_tmax_forecast schema
        truth_df: DataFrame with daily_tmax schema
        min_coverage_hours: Minimum coverage_hours for truth (default 18)

    Returns:
        DataFrame with forecast and truth joined, plus residual column
    """
    # Filter truth by coverage quality
    truth_filtered = truth_df[
        truth_df["coverage_hours"] >= min_coverage_hours
    ].copy()

    # Normalize date columns for join
    # forecast has target_date_local, truth has date_local
    forecast_df = forecast_df.copy()
    truth_filtered = truth_filtered.copy()

    # Ensure both date columns are datetime for proper join
    forecast_df["target_date_local"] = pd.to_datetime(
        forecast_df["target_date_local"]
    ).dt.normalize()
    truth_filtered["date_local"] = pd.to_datetime(
        truth_filtered["date_local"]
    ).dt.normalize()

    # Select columns from truth to merge
    truth_cols = ["station_id", "date_local", "tmax_f", "qc_flags"]
    truth_for_merge = truth_filtered[truth_cols].copy()
    truth_for_merge = truth_for_merge.rename(
        columns={
            "tmax_f": "tmax_actual_f",
            "qc_flags": "truth_qc_flags",
        }
    )

    # Merge on station_id and matching dates
    merged = forecast_df.merge(
        truth_for_merge,
        left_on=["station_id", "target_date_local"],
        right_on=["station_id", "date_local"],
        how="inner",
    )

    # Drop redundant date_local column from truth
    if "date_local" in merged.columns:
        merged = merged.drop(columns=["date_local"])

    # Compute residual (forecast - observed)
    # Positive residual = forecast too warm
    merged["residual"] = merged["tmax_pred_f"] - merged["tmax_actual_f"]

    return merged


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add seasonal and calendar encoding features.

    Features added:
    - sin_doy, cos_doy: Sinusoidal encoding of day-of-year
    - month: Integer month (1-12)

    These capture systematic seasonal patterns in forecast bias.

    Args:
        df: DataFrame with target_date_local column

    Returns:
        DataFrame with seasonal feature columns added
    """
    df = df.copy()

    # Extract day-of-year for seasonal encoding
    doy = df["target_date_local"].dt.dayofyear

    # Sinusoidal encoding (accounts for year wraparound)
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    # Month for coarse seasonal regime
    df["month"] = df["target_date_local"].dt.month

    return df


def add_forecast_source(
    df: pd.DataFrame,
    source_col: str = "source",
    output_col: str = "forecast_source",
) -> pd.DataFrame:
    """Add forecast_source column from source column.

    Args:
        df: DataFrame with source column
        source_col: Name of source column (default "source")
        output_col: Name of output column (default "forecast_source")

    Returns:
        DataFrame with forecast_source column added
    """
    df = df.copy()
    df[output_col] = df[source_col]
    return df


def build_train_daily_tmax(
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    min_coverage_hours: int = 18,
    drop_warmup_nulls: bool = False,
    validate: bool = True,
) -> pd.DataFrame:
    """Build complete training dataset from forecast and truth data.

    Pipeline steps:
    1. Join forecast to truth (filters low-quality days)
    2. Sort by issue_time_utc for correct rolling computation
    3. Add seasonal features (sin_doy, cos_doy, month)
    4. Add forecast_source column
    5. Compute rolling bias/error statistics
    6. Select and order output columns
    7. Optionally validate against schema

    Args:
        forecast_df: DataFrame with daily_tmax_forecast schema
        truth_df: DataFrame with daily_tmax schema
        min_coverage_hours: Minimum coverage for truth data (default 18)
        drop_warmup_nulls: If True, drop rows with NaN in rolling features
        validate: If True, validate output schema (default True)

    Returns:
        DataFrame with train_daily_tmax schema
    """
    # Step 1: Join forecast to truth
    df = join_forecast_to_truth(forecast_df, truth_df, min_coverage_hours)

    if df.empty:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=TRAIN_DAILY_TMAX_FIELDS)

    # Step 2: Sort by issue_time for correct rolling computation
    df = df.sort_values(["station_id", "lead_hours", "issue_time_utc"])
    df = df.reset_index(drop=True)

    # Step 3: Add seasonal features
    df = add_seasonal_features(df)

    # Step 4: Add forecast_source
    df = add_forecast_source(df)

    # Step 5: Compute rolling features
    # Groups by (station_id, lead_hours) for lead-specific statistics
    df = compute_all_rolling_features(
        df,
        residual_col="residual",
        bias_windows=[7, 14, 30],
        rmse_windows=[14, 30],
        group_cols=["station_id", "lead_hours"],
    )

    # Step 6: Select and order output columns
    output_cols = TRAIN_DAILY_TMAX_FIELDS.copy()
    df = df[output_cols]

    # Optionally drop warm-up rows with NaN rolling features
    if drop_warmup_nulls:
        rolling_cols = ["bias_7d", "bias_14d", "bias_30d", "rmse_14d", "rmse_30d", "sigma_lead"]
        df = df.dropna(subset=rolling_cols)
        df = df.reset_index(drop=True)

    # Step 7: Validate
    if validate:
        validate_train_daily_tmax(df, allow_warmup_nulls=not drop_warmup_nulls)

    return df


def write_train_daily_tmax(
    train_df: pd.DataFrame,
    output_path: Path | str,
    validate: bool = True,
) -> Path:
    """Validate and write training dataset to parquet.

    Args:
        train_df: DataFrame with train_daily_tmax schema
        output_path: Path to write parquet file
        validate: If True, validate before writing (default True)

    Returns:
        Path to written output file

    Raises:
        ValueError: If validation fails
    """
    output_path = Path(output_path)

    if validate:
        validate_train_daily_tmax(train_df)

    # Atomic write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    train_df.to_parquet(tmp_path, index=False)
    tmp_path.rename(output_path)

    print(f"[features] wrote {len(train_df)} rows to {output_path}")
    return output_path
