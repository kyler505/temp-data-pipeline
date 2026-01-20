"""Metrics slicing by various dimensions.

Compute metrics broken down by:
- Month (1-12)
- Season (DJF, MAM, JJA, SON)
- Lead hour bucket
- Temperature regime (cold, normal, hot)

No trading metrics - pure temperature evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tempdata.eval.metrics import compute_forecast_metrics, ForecastMetrics


def compute_metrics_by_slice(
    predictions_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute metrics broken down by various slices.

    Args:
        predictions_df: DataFrame with:
            - y_pred_f, y_true_f: Predictions and actuals
            - Optional: month, lead_hours, target_date_local

    Returns:
        Dictionary of slice name -> {slice_value -> metrics dict}
    """
    slices = {}

    # By month
    if "month" in predictions_df.columns:
        slices["by_month"] = _slice_by_column(predictions_df, "month")

    # By season
    if "month" in predictions_df.columns:
        slices["by_season"] = _slice_by_season(predictions_df)

    # By lead hour bucket
    if "lead_hours" in predictions_df.columns:
        slices["by_lead_bucket"] = _slice_by_lead_bucket(predictions_df)

    # By temperature regime
    if "y_true_f" in predictions_df.columns:
        slices["by_temp_regime"] = _slice_by_temp_regime(predictions_df)

    return slices


def _slice_by_column(
    df: pd.DataFrame,
    column: str,
) -> dict[str, dict[str, Any]]:
    """Compute metrics for each unique value of a column.

    Args:
        df: Predictions DataFrame
        column: Column to slice by

    Returns:
        Dictionary of column_value -> metrics dict
    """
    result = {}
    for value in sorted(df[column].unique()):
        subset = df[df[column] == value]
        if len(subset) >= 10:  # Minimum for stable metrics
            metrics = compute_forecast_metrics(subset)
            result[str(value)] = metrics.to_dict()
    return result


def _slice_by_season(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute metrics by meteorological season.

    Seasons:
    - DJF: December, January, February (Winter)
    - MAM: March, April, May (Spring)
    - JJA: June, July, August (Summer)
    - SON: September, October, November (Fall)
    """
    season_map = {
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    }

    df = df.copy()
    df["season"] = df["month"].map(season_map)

    result = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        subset = df[df["season"] == season]
        if len(subset) >= 10:
            metrics = compute_forecast_metrics(subset)
            result[season] = metrics.to_dict()

    return result


def _slice_by_lead_bucket(
    df: pd.DataFrame,
    buckets: list[tuple[int, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute metrics by lead hour bucket.

    Args:
        df: Predictions DataFrame with lead_hours column
        buckets: List of (low, high) buckets

    Returns:
        Dictionary of bucket_label -> metrics dict
    """
    if buckets is None:
        buckets = [(0, 24), (24, 48), (48, 72), (72, 120)]

    lead_hours = df["lead_hours"].values
    result = {}

    for lo, hi in buckets:
        mask = (lead_hours >= lo) & (lead_hours < hi)
        subset = df[mask]
        if len(subset) >= 10:
            label = f"{lo}-{hi}h"
            metrics = compute_forecast_metrics(subset)
            result[label] = metrics.to_dict()

    return result


def _slice_by_temp_regime(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute metrics by temperature regime.

    Regimes based on actual temperature quantiles:
    - cold: Below 25th percentile
    - normal: 25th to 75th percentile
    - hot: Above 75th percentile
    """
    temps = df["y_true_f"].values
    q25, q75 = np.percentile(temps, [25, 75])

    result = {}

    # Cold regime
    cold_mask = temps < q25
    if cold_mask.sum() >= 10:
        metrics = compute_forecast_metrics(df[cold_mask])
        result[f"cold (<{q25:.0f}°F)"] = metrics.to_dict()

    # Normal regime
    normal_mask = (temps >= q25) & (temps <= q75)
    if normal_mask.sum() >= 10:
        metrics = compute_forecast_metrics(df[normal_mask])
        result[f"normal ({q25:.0f}-{q75:.0f}°F)"] = metrics.to_dict()

    # Hot regime
    hot_mask = temps > q75
    if hot_mask.sum() >= 10:
        metrics = compute_forecast_metrics(df[hot_mask])
        result[f"hot (>{q75:.0f}°F)"] = metrics.to_dict()

    return result
