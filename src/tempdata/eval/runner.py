"""Evaluation runner - main orchestration for temperature evaluation.

This module provides the main entry point for running evaluations:
1. Load config
2. Load and filter data
3. Split train/val/test
4. Fit model and uncertainty
5. Generate predictions
6. Compute metrics
7. Write artifacts

No trading logic - pure temperature evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tempdata.eval.config import EvalConfig


@dataclass
class EvalResult:
    """Result container for an evaluation run.

    Attributes:
        run_id: Unique run identifier
        config: Evaluation configuration used
        predictions_df: DataFrame with predictions
        metrics: Computed evaluation metrics
        artifacts: Dictionary of artifact paths
    """
    run_id: str
    config: EvalConfig
    predictions_df: pd.DataFrame
    metrics: "EvalMetrics"
    artifacts: dict[str, Path]


def run_evaluation(
    config: EvalConfig,
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
    run_id: str | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> EvalResult:
    """Run a complete temperature evaluation.

    This is the main entry point for the evaluation framework.

    Args:
        config: Evaluation configuration
        forecast_df: Forecast data with tmax_pred_f
        truth_df: Truth data with tmax_f
        feature_df: Optional pre-built features
        run_id: Optional run identifier (auto-generated if not provided)
        output_dir: Optional output directory
        verbose: Whether to print progress

    Returns:
        EvalResult with predictions, metrics, and artifact paths
    """
    from tempdata.eval.config import generate_run_id
    from tempdata.eval.data import load_eval_data, print_data_summary
    from tempdata.eval.metrics import (
        compute_forecast_metrics,
        compute_calibration_metrics,
        EvalMetrics,
        print_metrics_summary,
    )
    from tempdata.eval.models import create_forecaster
    from tempdata.eval.report import write_all_artifacts
    from tempdata.eval.slicing import compute_metrics_by_slice
    from tempdata.eval.uncertainty import create_uncertainty_model

    # Generate run ID if not provided
    if run_id is None:
        run_id = generate_run_id()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"DAILY TMAX EVALUATION: {run_id}")
        print(f"{'=' * 60}")
        print(f"Run name: {config.run_name}")
        print(f"Stations: {', '.join(config.station_ids)}")
        print(f"Date range: {config.start_date_local} to {config.end_date_local}")
        print()

    # Step 1: Load and prepare data
    if verbose:
        print("[eval] Loading data...")

    dataset = load_eval_data(
        config=config,
        forecast_df=forecast_df,
        truth_df=truth_df,
        feature_df=feature_df,
    )

    if verbose:
        print_data_summary(dataset)

    # Step 2: Create and fit model
    if verbose:
        print("\n[eval] Fitting model...")

    forecaster = create_forecaster(
        model_type=config.model.type,
        alpha=config.model.alpha,
        features=config.model.features,
    )
    forecaster.fit(dataset.train)

    # Step 3: Generate predictions on test set
    if verbose:
        print("[eval] Generating predictions...")

    predictions_df = _generate_predictions(
        df=dataset.test,
        forecaster=forecaster,
        config=config,
        df_train=dataset.train,
    )

    # Step 4: Compute metrics
    if verbose:
        print("[eval] Computing metrics...")

    forecast_metrics = compute_forecast_metrics(predictions_df)

    # Calibration metrics if sigma is available
    calibration_metrics = None
    if "y_pred_sigma_f" in predictions_df.columns:
        calibration_metrics = compute_calibration_metrics(predictions_df)

    # Sliced metrics
    slices = compute_metrics_by_slice(predictions_df)

    metrics = EvalMetrics(
        forecast=forecast_metrics,
        calibration=calibration_metrics,
        slices=slices,
    )

    if verbose:
        print_metrics_summary(metrics)

    # Step 5: Write artifacts
    if verbose:
        print("[eval] Writing artifacts...")

    artifacts = write_all_artifacts(
        config=config,
        metrics=metrics,
        predictions_df=predictions_df,
        run_id=run_id,
        base_path=output_dir,
    )

    if verbose:
        print(f"\nRun complete: {run_id}")
        print(f"Artifacts: {artifacts['config'].parent}")

    return EvalResult(
        run_id=run_id,
        config=config,
        predictions_df=predictions_df,
        metrics=metrics,
        artifacts=artifacts,
    )


def _generate_predictions(
    df: pd.DataFrame,
    forecaster,
    config: EvalConfig,
    df_train: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions DataFrame with all required columns.

    Args:
        df: Test DataFrame
        forecaster: Fitted forecaster
        config: Evaluation configuration
        df_train: Training DataFrame (for uncertainty fitting)

    Returns:
        DataFrame with y_pred_f, y_true_f, and metadata
    """
    from tempdata.eval.uncertainty import create_uncertainty_model

    predictions_df = df.copy()

    # Point predictions
    predictions_df["y_pred_f"] = forecaster.predict_mu(df)
    predictions_df["y_true_f"] = df["tmax_actual_f"].values

    # Uncertainty estimates
    uncertainty_model = create_uncertainty_model(
        sigma_type=config.uncertainty.type,
        sigma_buckets=config.uncertainty.buckets,
        sigma_floor=config.uncertainty.sigma_floor,
    )

    # Fit on training residuals
    train_mu = forecaster.predict_mu(df_train)
    train_residuals = train_mu - df_train["tmax_actual_f"].values
    uncertainty_model.fit(df_train, train_residuals)

    # Predict sigma
    predictions_df["y_pred_sigma_f"] = uncertainty_model.predict_sigma(df)

    # Add metadata columns
    if "lead_hours" in df.columns:
        predictions_df["lead_hours"] = df["lead_hours"]
    if "month" not in predictions_df.columns and "target_date_local" in df.columns:
        predictions_df["month"] = pd.to_datetime(df["target_date_local"]).dt.month
    if "doy" not in predictions_df.columns and "target_date_local" in df.columns:
        predictions_df["doy"] = pd.to_datetime(df["target_date_local"]).dt.dayofyear

    return predictions_df
