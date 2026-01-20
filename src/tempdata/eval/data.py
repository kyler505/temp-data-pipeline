"""Data loading and preparation for temperature evaluation.

This module provides data loading with QC filtering for evaluation runs.
No trading/market concepts - pure temperature data handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tempdata.eval.config import EvalConfig


@dataclass
class EvalDataset:
    """Container for evaluation data splits.

    Attributes:
        train: Training data
        val: Validation data (may be empty for walk_forward)
        test: Test data
        full: Complete dataset before splitting
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    full: pd.DataFrame

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.train)

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return len(self.val)

    @property
    def n_test(self) -> int:
        """Number of test samples."""
        return len(self.test)


def load_eval_data(
    config: EvalConfig,
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
) -> EvalDataset:
    """Load and prepare data for temperature evaluation.

    Joins forecast and truth data, applies QC filters, and creates
    train/val/test splits according to configuration.

    Args:
        config: Evaluation configuration
        forecast_df: DataFrame with forecast data (must have tmax_pred_f)
        truth_df: DataFrame with truth/observed data (must have tmax_f)
        feature_df: Optional pre-built feature DataFrame

    Returns:
        EvalDataset with train/val/test splits
    """
    # If feature_df provided, use it directly
    if feature_df is not None:
        df = feature_df.copy()
    else:
        # Join forecast and truth data
        df = _join_forecast_truth(forecast_df, truth_df)

    # Apply filters
    df = _apply_filters(df, config)

    # Add engineered features if not present
    df = _ensure_features(df)

    # Sort by time for proper splitting
    df = df.sort_values(["station_id", "target_date_local"]).reset_index(drop=True)

    # Create splits
    from tempdata.eval.splits import create_split

    splitter = create_split(config.split)
    train_df, val_df, test_df = splitter.split(df)

    return EvalDataset(
        train=train_df,
        val=val_df,
        test=test_df,
        full=df,
    )


def _join_forecast_truth(
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join forecast and truth data on station + date.

    Args:
        forecast_df: Forecast data with columns:
            - station_id, target_date_local, tmax_pred_f, lead_hours, etc.
        truth_df: Truth data with columns:
            - station_id, date_local, tmax_f, coverage_hours, etc.

    Returns:
        Joined DataFrame with both forecast and truth columns
    """
    # Standardize column names
    forecast_df = forecast_df.copy()
    truth_df = truth_df.copy()

    # Ensure date columns are date type
    if "target_date_local" in forecast_df.columns:
        forecast_df["target_date_local"] = pd.to_datetime(
            forecast_df["target_date_local"]
        ).dt.date

    if "date_local" in truth_df.columns:
        truth_df["target_date_local"] = pd.to_datetime(truth_df["date_local"]).dt.date
    elif "target_date_local" not in truth_df.columns:
        raise ValueError("truth_df must have date_local or target_date_local column")

    # Rename truth column if needed
    if "tmax_f" in truth_df.columns and "tmax_actual_f" not in truth_df.columns:
        truth_df["tmax_actual_f"] = truth_df["tmax_f"]

    # Select truth columns for join
    truth_cols = ["station_id", "target_date_local", "tmax_actual_f"]
    if "coverage_hours" in truth_df.columns:
        truth_cols.append("coverage_hours")
    if "qc_flags" in truth_df.columns:
        truth_cols.append("qc_flags")

    truth_subset = truth_df[truth_cols].drop_duplicates()

    # Join on station + date
    df = forecast_df.merge(
        truth_subset,
        on=["station_id", "target_date_local"],
        how="inner",
    )

    return df


def _apply_filters(df: pd.DataFrame, config: EvalConfig) -> pd.DataFrame:
    """Apply QC and configuration filters.

    Args:
        df: Input DataFrame
        config: Evaluation configuration

    Returns:
        Filtered DataFrame
    """
    original_len = len(df)

    # Filter by station
    df = df[df["station_id"].isin(config.station_ids)]

    # Filter by date range
    df = df[
        (df["target_date_local"] >= config.start_date_local) &
        (df["target_date_local"] <= config.end_date_local)
    ]

    # Filter by coverage hours if column exists
    if "coverage_hours" in df.columns:
        df = df[df["coverage_hours"] >= config.min_coverage_hours]

    # Filter by lead hours if specified
    if config.lead_hours_allowed and "lead_hours" in df.columns:
        df = df[df["lead_hours"].isin(config.lead_hours_allowed)]

    # Remove rows with NaN in key columns
    required_cols = ["tmax_pred_f", "tmax_actual_f"]
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    filtered_len = len(df)
    if filtered_len == 0:
        raise ValueError(
            f"No data remaining after filters. "
            f"Started with {original_len} rows, ended with 0."
        )

    return df.reset_index(drop=True)


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features if not present.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # Day of year features
    if "sin_doy" not in df.columns or "cos_doy" not in df.columns:
        if "target_date_local" in df.columns:
            doy = pd.to_datetime(df["target_date_local"]).dt.dayofyear
            df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
            df["cos_doy"] = np.cos(2 * np.pi * doy / 365)
            df["doy"] = doy

    # Month feature
    if "month" not in df.columns and "target_date_local" in df.columns:
        df["month"] = pd.to_datetime(df["target_date_local"]).dt.month

    # Rolling bias features (if not present, fill with 0)
    for col in ["bias_7d", "bias_14d", "rmse_30d", "sigma_lead"]:
        if col not in df.columns:
            df[col] = 0.0

    # Sort by station and date to ensure correct rolling/lag calculations
    if "station_id" in df.columns and "target_date_local" in df.columns:
        df = df.sort_values(["station_id", "target_date_local"])

    # Lag features for persistence
    if "tmax_actual_f_lag1" not in df.columns:
        if "tmax_actual_f" in df.columns and "station_id" in df.columns:
            df["tmax_actual_f_lag1"] = df.groupby("station_id")["tmax_actual_f"].shift(1)
        # Fill NaN at the beginning with first available or 0?
        # For persistence, we can backfill or leave NaNs (which will be handled/filled later)
        # PersistenceForecaster fills na with 0.0.

    return df


def print_data_summary(dataset: EvalDataset) -> None:
    """Print a summary of the evaluation dataset.

    Args:
        dataset: EvalDataset to summarize
    """
    print("\n" + "=" * 60)
    print("EVALUATION DATA SUMMARY")
    print("=" * 60)

    print(f"\nTotal samples:    {len(dataset.full):,}")
    print(f"Training samples: {dataset.n_train:,} ({100 * dataset.n_train / len(dataset.full):.1f}%)")
    print(f"Validation:       {dataset.n_val:,} ({100 * dataset.n_val / len(dataset.full):.1f}%)")
    print(f"Test samples:     {dataset.n_test:,} ({100 * dataset.n_test / len(dataset.full):.1f}%)")

    # Date ranges
    if "target_date_local" in dataset.full.columns:
        print(f"\nDate range: {dataset.full['target_date_local'].min()} to {dataset.full['target_date_local'].max()}")

    # Stations
    if "station_id" in dataset.full.columns:
        stations = dataset.full["station_id"].unique()
        print(f"Stations: {', '.join(stations)}")

    # Feature columns
    feature_cols = [c for c in dataset.train.columns if c not in [
        "station_id", "target_date_local", "issue_time_utc",
        "tmax_actual_f", "coverage_hours", "qc_flags"
    ]]
    print(f"\nFeature columns ({len(feature_cols)}): {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}")
