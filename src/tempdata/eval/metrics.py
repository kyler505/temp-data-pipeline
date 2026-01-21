"""Metrics computation for temperature evaluation.

This module provides:
1. Point metrics: MAE, RMSE, bias
2. Calibration metrics: interval coverage, sharpness
3. Optional probabilistic scores: pinball loss

No trading metrics (PnL, Sharpe, etc.) - pure temperature evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ForecastMetrics:
    """Point forecast accuracy metrics.

    Attributes:
        n_samples: Number of samples evaluated
        mae: Mean Absolute Error (°F)
        rmse: Root Mean Squared Error (°F)
        bias: Mean error, predicted - actual (°F)
        std_error: Standard deviation of errors (°F)
    """
    n_samples: int
    mae: float
    rmse: float
    bias: float
    std_error: float
    r2: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "n_samples": self.n_samples,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "bias": round(self.bias, 4),
            "std_error": round(self.std_error, 4),
        }
        if self.r2 is not None:
            d["r2"] = round(self.r2, 4)
        return d


@dataclass
class CalibrationMetrics:
    """Calibration and uncertainty metrics.

    Attributes:
        coverage_50: Fraction of actuals within 50% prediction interval
        coverage_80: Fraction of actuals within 80% prediction interval
        coverage_90: Fraction of actuals within 90% prediction interval
        mean_sigma: Mean predicted standard deviation (°F)
        sharpness_50: Mean width of 50% prediction interval (°F)
        sharpness_80: Mean width of 80% prediction interval (°F)
        sharpness_90: Mean width of 90% prediction interval (°F)
    """
    coverage_50: float
    coverage_80: float
    coverage_90: float
    mean_sigma: float
    sharpness_50: float
    sharpness_80: float
    sharpness_90: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "coverage_50": round(self.coverage_50, 4),
            "coverage_80": round(self.coverage_80, 4),
            "coverage_90": round(self.coverage_90, 4),
            "mean_sigma": round(self.mean_sigma, 4),
            "sharpness_50": round(self.sharpness_50, 4),
            "sharpness_80": round(self.sharpness_80, 4),
            "sharpness_90": round(self.sharpness_90, 4),
        }


@dataclass
class EvalMetrics:
    """Complete evaluation metrics summary.

    Contains both forecast and calibration metrics, plus sliced breakdowns.
    """
    forecast: ForecastMetrics
    calibration: CalibrationMetrics | None = None
    slices: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "forecast": self.forecast.to_dict(),
        }
        if self.calibration is not None:
            result["calibration"] = self.calibration.to_dict()
        if self.slices:
            result["slices"] = self.slices
        return result


def compute_forecast_metrics(predictions_df: pd.DataFrame) -> ForecastMetrics:
    """Compute point forecast accuracy metrics.

    Args:
        predictions_df: DataFrame with columns:
            - y_pred_f: Predicted temperature (mu)
            - y_true_f: Actual temperature

    Returns:
        ForecastMetrics object
    """
    y_true = predictions_df["y_true_f"].values
    y_pred = predictions_df["y_pred_f"].values

    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    # R2 calculation
    ss_res = np.sum(sq_errors)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return ForecastMetrics(
        n_samples=len(y_true),
        mae=float(np.mean(abs_errors)),
        rmse=float(np.sqrt(np.mean(sq_errors))),
        bias=float(np.mean(errors)),
        std_error=float(np.std(errors)),
        r2=float(r2),
    )


def compute_calibration_metrics(
    predictions_df: pd.DataFrame,
    coverage_levels: list[float] | None = None,
) -> CalibrationMetrics:
    """Compute calibration/uncertainty metrics.

    Calculates interval coverage and sharpness for Gaussian prediction intervals.

    Args:
        predictions_df: DataFrame with columns:
            - y_pred_f: Predicted temperature (mu)
            - y_true_f: Actual temperature
            - y_pred_sigma_f: Predicted standard deviation (sigma)

    Returns:
        CalibrationMetrics object
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("scipy required for calibration metrics")

    if coverage_levels is None:
        coverage_levels = [0.50, 0.80, 0.90]

    y_true = predictions_df["y_true_f"].values
    y_pred = predictions_df["y_pred_f"].values
    sigma = predictions_df["y_pred_sigma_f"].values

    coverages = {}
    sharpnesses = {}

    for level in coverage_levels:
        # Z-score for this coverage level
        alpha = 1 - level
        z = stats.norm.ppf(1 - alpha / 2)

        # Interval bounds
        lower = y_pred - z * sigma
        upper = y_pred + z * sigma

        # Coverage: fraction of actuals within interval
        in_interval = (y_true >= lower) & (y_true <= upper)
        coverages[level] = float(np.mean(in_interval))

        # Sharpness: mean interval width
        sharpnesses[level] = float(np.mean(upper - lower))

    return CalibrationMetrics(
        coverage_50=coverages.get(0.50, 0.0),
        coverage_80=coverages.get(0.80, 0.0),
        coverage_90=coverages.get(0.90, 0.0),
        mean_sigma=float(np.mean(sigma)),
        sharpness_50=sharpnesses.get(0.50, 0.0),
        sharpness_80=sharpnesses.get(0.80, 0.0),
        sharpness_90=sharpnesses.get(0.90, 0.0),
    )


def compute_pinball_loss(
    predictions_df: pd.DataFrame,
    quantiles: list[float] | None = None,
) -> dict[float, float]:
    """Compute pinball loss for quantile predictions.

    Args:
        predictions_df: DataFrame with columns:
            - y_true_f: Actual temperature
            - q{XX}: Quantile predictions (e.g., q10, q50, q90)

    Returns:
        Dictionary mapping quantile to pinball loss
    """
    if quantiles is None:
        quantiles = [0.10, 0.50, 0.90]

    y_true = predictions_df["y_true_f"].values
    losses = {}

    for q in quantiles:
        col = f"q{int(q * 100)}"
        if col not in predictions_df.columns:
            continue

        y_q = predictions_df[col].values
        error = y_true - y_q

        # Pinball loss: q * max(error, 0) + (1-q) * max(-error, 0)
        loss = np.where(
            error >= 0,
            q * error,
            (q - 1) * error
        )
        losses[q] = float(np.mean(loss))

    return losses


def print_metrics_summary(metrics: EvalMetrics) -> None:
    """Print a formatted summary of evaluation metrics.

    Args:
        metrics: EvalMetrics object to summarize
    """
    print("\n" + "=" * 60)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 60)

    fm = metrics.forecast
    print("\n--- FORECAST PERFORMANCE ---")
    print(f"  Samples:     {fm.n_samples:,}")
    print(f"  MAE:         {fm.mae:.2f}°F")
    print(f"  RMSE:        {fm.rmse:.2f}°F")
    print(f"  Bias:        {fm.bias:+.2f}°F")
    print(f"  Std Error:   {fm.std_error:.2f}°F")

    if metrics.calibration is not None:
        cm = metrics.calibration
        print("\n--- CALIBRATION ---")
        print(f"  Mean σ:      {cm.mean_sigma:.2f}°F")
        print(f"  50% PI cov:  {100 * cm.coverage_50:.1f}% (target: 50%)")
        print(f"  80% PI cov:  {100 * cm.coverage_80:.1f}% (target: 80%)")
        print(f"  90% PI cov:  {100 * cm.coverage_90:.1f}% (target: 90%)")
        print(f"  90% width:   {cm.sharpness_90:.1f}°F")

    print()
