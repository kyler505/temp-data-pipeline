"""Rolling statistics for forecast bias and error computation.

All rolling features use .shift(1) to ensure strict causality:
- No information from the current row is used in its own features
- Only past residuals contribute to rolling statistics

This prevents target leakage and ensures features are valid at forecast time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling_bias(
    df: pd.DataFrame,
    residual_col: str = "residual",
    windows: list[int] | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute rolling mean bias over specified windows.

    Bias is defined as (forecast - observed), so positive bias means
    the forecast runs warm.

    Uses .shift(1) to ensure no lookahead: the current row's residual
    is excluded from its own rolling statistics.

    Args:
        df: DataFrame with residual column, must be sorted by time
        residual_col: Name of the residual column (forecast - observed)
        windows: List of window sizes in days (default [7, 14, 30])
        group_cols: Columns to group by (default ["station_id", "lead_hours"])

    Returns:
        DataFrame with added bias_{w}d columns for each window
    """
    if windows is None:
        windows = [7, 14, 30]
    if group_cols is None:
        group_cols = ["station_id", "lead_hours"]

    df = df.copy()

    for w in windows:
        col_name = f"bias_{w}d"
        df[col_name] = (
            df.groupby(group_cols, group_keys=False)[residual_col]
            .apply(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    return df


def compute_rolling_rmse(
    df: pd.DataFrame,
    residual_col: str = "residual",
    windows: list[int] | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute rolling RMSE over specified windows.

    RMSE = sqrt(mean(residual^2)) over the window.

    Uses .shift(1) to ensure no lookahead: the current row's residual
    is excluded from its own rolling statistics.

    Args:
        df: DataFrame with residual column, must be sorted by time
        residual_col: Name of the residual column (forecast - observed)
        windows: List of window sizes in days (default [14, 30])
        group_cols: Columns to group by (default ["station_id", "lead_hours"])

    Returns:
        DataFrame with added rmse_{w}d columns for each window
    """
    if windows is None:
        windows = [14, 30]
    if group_cols is None:
        group_cols = ["station_id", "lead_hours"]

    df = df.copy()

    # Compute squared residuals for RMSE calculation
    residual_sq = df[residual_col] ** 2

    for w in windows:
        col_name = f"rmse_{w}d"
        # Compute rolling mean of squared residuals, then sqrt
        rolling_mse = (
            df.groupby(group_cols, group_keys=False)[residual_col]
            .apply(
                lambda x: (x.shift(1) ** 2).rolling(w, min_periods=1).mean()
            )
        )
        df[col_name] = np.sqrt(rolling_mse)

    return df


def compute_sigma_lead(
    df: pd.DataFrame,
    residual_col: str = "residual",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute historical residual standard deviation per lead_hours.

    This captures the typical uncertainty for a given forecast horizon,
    computed from all historical data up to (but not including) the
    current row.

    Uses expanding window with .shift(1) to maintain causality.

    Args:
        df: DataFrame with residual column, must be sorted by time
        residual_col: Name of the residual column (forecast - observed)
        group_cols: Columns to group by (default ["station_id", "lead_hours"])

    Returns:
        DataFrame with added sigma_lead column
    """
    if group_cols is None:
        group_cols = ["station_id", "lead_hours"]

    df = df.copy()

    # Use expanding window for historical std dev
    # shift(1) ensures current row is excluded
    df["sigma_lead"] = (
        df.groupby(group_cols, group_keys=False)[residual_col]
        .apply(lambda x: x.shift(1).expanding(min_periods=2).std())
    )

    return df


def compute_all_rolling_features(
    df: pd.DataFrame,
    residual_col: str = "residual",
    bias_windows: list[int] | None = None,
    rmse_windows: list[int] | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute all rolling bias, RMSE, and sigma_lead features.

    Convenience function that applies all rolling statistics in sequence.

    Args:
        df: DataFrame with residual column, must be sorted by time
        residual_col: Name of the residual column (forecast - observed)
        bias_windows: Window sizes for bias (default [7, 14, 30])
        rmse_windows: Window sizes for RMSE (default [14, 30])
        group_cols: Columns to group by (default ["station_id", "lead_hours"])

    Returns:
        DataFrame with all rolling feature columns added
    """
    if bias_windows is None:
        bias_windows = [7, 14, 30]
    if rmse_windows is None:
        rmse_windows = [14, 30]
    if group_cols is None:
        group_cols = ["station_id", "lead_hours"]

    df = compute_rolling_bias(df, residual_col, bias_windows, group_cols)
    df = compute_rolling_rmse(df, residual_col, rmse_windows, group_cols)
    df = compute_sigma_lead(df, residual_col, group_cols)

    return df
