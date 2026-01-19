"""Data loading, joining, and QC filtering for backtesting.

This module handles:
1. Loading forecasts and truth data
2. Joining on (station_id, target_date_local)
3. Applying QC filters (coverage, flags)
4. Time-based train/val/test splitting
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tempdata.features.build_train_daily_tmax import (
    add_forecast_source,
    add_seasonal_features,
    join_forecast_to_truth,
)
from tempdata.features.rolling_stats import compute_all_rolling_features
from tempdata.schemas.qc_flags import QC_INCOMPLETE_DAY

if TYPE_CHECKING:
    from tempdata.backtest.config import BacktestConfig


@dataclass
class BacktestDataset:
    """Container for split backtest data.
    
    Attributes:
        full: Complete dataset after joining and filtering
        train: Training split (earliest data)
        val: Validation split (middle data)
        test: Test split (latest data)
        stats: Summary statistics about the data
    """
    full: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    stats: dict


def load_backtest_data(
    config: BacktestConfig,
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
) -> BacktestDataset:
    """Load and prepare data for backtesting.
    
    If feature_df is provided, uses it directly (assumes pre-joined).
    Otherwise, joins forecast_df to truth_df and computes features.
    
    Args:
        config: Backtest configuration
        forecast_df: DataFrame with daily_tmax_forecast schema
        truth_df: DataFrame with daily_tmax schema
        feature_df: Optional pre-built feature DataFrame
        
    Returns:
        BacktestDataset with train/val/test splits
        
    Raises:
        ValueError: If no data remains after filtering
    """
    # Track statistics
    stats: dict = {}
    
    if feature_df is not None:
        # Use pre-built features
        df = feature_df.copy()
        stats["source"] = "prebuilt_features"
        stats["rows_input"] = len(df)
    else:
        # Join forecasts to truth
        stats["source"] = "joined"
        stats["forecast_rows_input"] = len(forecast_df)
        stats["truth_rows_input"] = len(truth_df)
        
        df = _join_and_build_features(
            forecast_df, truth_df, config.min_coverage_hours
        )
        stats["rows_after_join"] = len(df)
    
    if df.empty:
        raise ValueError(
            "No data after join. Check date ranges overlap and station_id matches."
        )
    
    # Apply filters
    df = _apply_filters(df, config)
    stats["rows_after_filters"] = len(df)
    
    if df.empty:
        raise ValueError(
            "No data after filtering. Check lead_hours_allowed and date range."
        )
    
    # Compute date range stats
    stats["min_target_date"] = df["target_date_local"].min()
    stats["max_target_date"] = df["target_date_local"].max()
    stats["stations"] = sorted(df["station_id"].unique().tolist())
    stats["lead_hours_present"] = sorted(df["lead_hours"].unique().tolist())
    
    # Sort by target_date for proper time-based splitting
    df = df.sort_values(["station_id", "target_date_local", "issue_time_utc"])
    df = df.reset_index(drop=True)
    
    # Perform time-based split
    train_df, val_df, test_df = _time_split(df, config.train_frac, config.val_frac)
    
    stats["train_rows"] = len(train_df)
    stats["val_rows"] = len(val_df)
    stats["test_rows"] = len(test_df)
    
    if len(train_df) == 0:
        raise ValueError("Training set is empty after split.")
    
    return BacktestDataset(
        full=df,
        train=train_df,
        val=val_df,
        test=test_df,
        stats=stats,
    )


def _join_and_build_features(
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    min_coverage_hours: int,
) -> pd.DataFrame:
    """Join forecasts to truth and build features.
    
    Reuses logic from build_train_daily_tmax module.
    """
    # Join forecast to truth
    df = join_forecast_to_truth(forecast_df, truth_df, min_coverage_hours)
    
    if df.empty:
        return df
    
    # Sort by issue_time for correct rolling computation
    df = df.sort_values(["station_id", "lead_hours", "issue_time_utc"])
    df = df.reset_index(drop=True)
    
    # Add seasonal features
    df = add_seasonal_features(df)
    
    # Add forecast_source if not present
    if "forecast_source" not in df.columns and "source" in df.columns:
        df = add_forecast_source(df)
    
    # Compute rolling features
    df = compute_all_rolling_features(
        df,
        residual_col="residual",
        bias_windows=[7, 14, 30],
        rmse_windows=[14, 30],
        group_cols=["station_id", "lead_hours"],
    )
    
    return df


def _apply_filters(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Apply configuration-based filters to the data."""
    # Filter by station
    df = df[df["station_id"].isin(config.station_ids)].copy()
    
    # Filter by date range
    # Normalize target_date_local to date for comparison
    target_dates = pd.to_datetime(df["target_date_local"]).dt.date
    df = df[
        (target_dates >= config.start_date_local) &
        (target_dates <= config.end_date_local)
    ]
    
    # Filter by lead hours if specified
    if config.lead_hours_allowed is not None:
        df = df[df["lead_hours"].isin(config.lead_hours_allowed)]
    
    # Filter by QC flags - exclude incomplete days
    if "truth_qc_flags" in df.columns:
        df = df[(df["truth_qc_flags"] & QC_INCOMPLETE_DAY) == 0]
    
    return df.reset_index(drop=True)


def _time_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by time, preserving temporal order.
    
    Uses target_date_local for splitting to ensure no temporal leakage.
    """
    # Get unique dates in sorted order
    dates = df["target_date_local"].drop_duplicates().sort_values()
    n_dates = len(dates)
    
    train_end_idx = int(n_dates * train_frac)
    val_end_idx = int(n_dates * (train_frac + val_frac))
    
    train_dates = set(dates.iloc[:train_end_idx])
    val_dates = set(dates.iloc[train_end_idx:val_end_idx])
    test_dates = set(dates.iloc[val_end_idx:])
    
    # Normalize for comparison
    target_dates = pd.to_datetime(df["target_date_local"])
    
    train_df = df[target_dates.isin(train_dates)].copy()
    val_df = df[target_dates.isin(val_dates)].copy()
    test_df = df[target_dates.isin(test_dates)].copy()
    
    return train_df, val_df, test_df


def print_data_summary(dataset: BacktestDataset) -> None:
    """Print a summary of the loaded data."""
    stats = dataset.stats
    
    print("\n" + "=" * 60)
    print("BACKTEST DATA SUMMARY")
    print("=" * 60)
    
    print(f"\nData source: {stats['source']}")
    
    if stats["source"] == "joined":
        print(f"  Forecast rows input: {stats['forecast_rows_input']:,}")
        print(f"  Truth rows input: {stats['truth_rows_input']:,}")
        print(f"  Rows after join: {stats['rows_after_join']:,}")
    else:
        print(f"  Rows input: {stats['rows_input']:,}")
    
    print(f"  Rows after filters: {stats['rows_after_filters']:,}")
    
    print(f"\nDate range:")
    print(f"  Min target date: {stats['min_target_date']}")
    print(f"  Max target date: {stats['max_target_date']}")
    
    print(f"\nStations: {stats['stations']}")
    print(f"Lead hours: {stats['lead_hours_present']}")
    
    print(f"\nSplit sizes:")
    print(f"  Train: {stats['train_rows']:,} rows ({stats['train_rows']/stats['rows_after_filters']*100:.1f}%)")
    print(f"  Val:   {stats['val_rows']:,} rows ({stats['val_rows']/stats['rows_after_filters']*100:.1f}%)")
    print(f"  Test:  {stats['test_rows']:,} rows ({stats['test_rows']/stats['rows_after_filters']*100:.1f}%)")
    
    print("=" * 60 + "\n")


def check_data_overlap(
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> dict:
    """Check date overlap between forecast and truth datasets.
    
    Use this to diagnose join issues before running a full backtest.
    
    Returns:
        Dictionary with overlap statistics
    """
    # Get date ranges
    forecast_dates = pd.to_datetime(forecast_df["target_date_local"]).dt.date
    truth_dates = pd.to_datetime(truth_df["date_local"]).dt.date
    
    forecast_min = forecast_dates.min()
    forecast_max = forecast_dates.max()
    truth_min = truth_dates.min()
    truth_max = truth_dates.max()
    
    # Find overlap
    overlap_start = max(forecast_min, truth_min)
    overlap_end = min(forecast_max, truth_max)
    
    if overlap_start > overlap_end:
        overlap_days = 0
    else:
        overlap_days = (overlap_end - overlap_start).days + 1
    
    # Check stations
    forecast_stations = set(forecast_df["station_id"].unique())
    truth_stations = set(truth_df["station_id"].unique())
    common_stations = forecast_stations & truth_stations
    
    return {
        "forecast_date_range": (forecast_min, forecast_max),
        "truth_date_range": (truth_min, truth_max),
        "overlap_range": (overlap_start, overlap_end) if overlap_days > 0 else None,
        "overlap_days": overlap_days,
        "forecast_stations": sorted(forecast_stations),
        "truth_stations": sorted(truth_stations),
        "common_stations": sorted(common_stations),
    }
