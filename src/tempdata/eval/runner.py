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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tempdata.eval.config import EvalConfig
    from tempdata.eval.metrics import EvalMetrics



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
    config: "EvalConfig"
    predictions_df: pd.DataFrame
    metrics: "EvalMetrics"
    artifacts: dict[str, Path]


@dataclass
class MultiModelEvalResult:
    """Result container for a multi-model evaluation run.

    Attributes:
        run_id: Unique run identifier
        run_path: Path to the run directory
        results: Dictionary mapping model names to their EvalResult
        comparison: Dictionary containing comparison summary metrics
    """
    run_id: str
    run_path: Path
    results: dict[str, EvalResult]
    comparison: dict[str, Any]


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

    # Step 2: Evaluate model (fit, predict, metrics)
    predictions_df, metrics = _evaluate_model(
        config=config,
        dataset=dataset,
        verbose=verbose,
    )

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


def _evaluate_model(
    config: "EvalConfig",
    dataset: Any,
    verbose: bool = True,
) -> tuple[pd.DataFrame, "EvalMetrics"]:
    """Internal helper to fit and evaluate a single model.

    Args:
        config: Evaluation configuration
        dataset: Prepared dataset (EvalDataset)
        verbose: Whether to print progress

    Returns:
        Tuple of (predictions_df, metrics)
    """
    from tempdata.eval.metrics import (
        EvalMetrics,
        compute_accuracy_bands,
        compute_calibration_metrics,
        compute_forecast_metrics,
        print_metrics_summary,
    )
    from tempdata.eval.models import create_forecaster
    from tempdata.eval.slicing import compute_metrics_by_slice

    # Create and fit model
    if verbose:
        print(f"\n[eval] Fitting model: {config.model.type}")

    forecaster = create_forecaster(
        model_type=config.model.type,
        alpha=config.model.alpha,
        features=config.model.features,
    )
    forecaster.fit(dataset.train)

    # Generate predictions on test set
    if verbose:
        print("[eval] Generating predictions...")

    predictions_df = _generate_predictions(
        df=dataset.test,
        forecaster=forecaster,
        config=config,
        df_train=dataset.train,
    )

    # Compute metrics
    if verbose:
        print("[eval] Computing metrics...")

    forecast_metrics = compute_forecast_metrics(predictions_df)

    # Calibration metrics if sigma is available
    calibration_metrics = None
    if "y_pred_sigma_f" in predictions_df.columns:
        try:
            calibration_metrics = compute_calibration_metrics(predictions_df)
        except ImportError:
            if verbose:
                print("[eval] Skipping calibration metrics: scipy not installed")

    # Accuracy band metrics
    accuracy_bands = compute_accuracy_bands(predictions_df)

    # Sliced metrics
    slices = compute_metrics_by_slice(predictions_df)

    metrics = EvalMetrics(
        forecast=forecast_metrics,
        calibration=calibration_metrics,
        accuracy_bands=accuracy_bands,
        slices=slices,
    )

    if verbose:
        print_metrics_summary(metrics)

    return predictions_df, metrics


def run_multi_model_evaluation(
    configs: dict[str, "EvalConfig"],
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
    run_id: str | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> MultiModelEvalResult:
    """Run a multi-model evaluation experiment.

    Evaluates multiple models using their respective configs and saves all results
    under a single run ID with a unified structure.

    Args:
        configs: Dictionary mapping model names to their EvalConfig
        forecast_df: Forecast data
        truth_df: Truth data
        feature_df: Optional features
        run_id: Optional run ID
        output_dir: Optional output directory base path
        verbose: Whether to print progress

    Returns:
        MultiModelEvalResult containing results for all models
    """
    from tempdata.eval.config import generate_run_id
    from tempdata.eval.data import load_eval_data
    from tempdata.eval.report import (
        create_run_dir,
        write_comparison_summary,
        write_model_artifacts,
        write_run_metadata,
    )

    # Generate single run ID for the whole experiment
    if run_id is None:
        run_id = generate_run_id()

    # Create top-level run directory
    full_output_dir = create_run_dir(run_id, base_path=output_dir)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MULTI-MODEL EVALUATION: {run_id}")
        print(f"Models: {', '.join(configs.keys())}")
        print(f"{'=' * 60}\n")

    # Write top-level metadata
    write_run_metadata(
        run_id=run_id,
        run_name=f"Multi-Model Eval {run_id}",
        base_path=output_dir,
        model_names=list(configs.keys()),
    )

    results = {}

    for name, config in configs.items():
        if verbose:
            print(f"\n--- Evaluating Model: {name} ---")

        # Load data for this config
        # Note: We load data per model to respect config-specific settings (e.g. features)
        dataset = load_eval_data(
            config=config,
            forecast_df=forecast_df,
            truth_df=truth_df,
            feature_df=feature_df,
        )

        # Evaluate model
        predictions_df, metrics = _evaluate_model(
            config=config,
            dataset=dataset,
            verbose=verbose,
        )

        # Write model-specific artifacts
        artifacts = write_model_artifacts(
            model_name=name,
            config=config,
            metrics=metrics,
            predictions_df=predictions_df,
            run_id=run_id,
            base_path=output_dir,
        )

        results[name] = EvalResult(
            run_id=run_id,
            config=config,
            predictions_df=predictions_df,
            metrics=metrics,
            artifacts=artifacts,
        )

    # Prepare results for comparison summary
    summary_data = {}
    for name, res in results.items():
        summary_data[name] = {
            "metrics": res.metrics.to_dict()
        }

    # Create and write comparison summary
    comparison = write_comparison_summary(
        run_id=run_id,
        model_results=summary_data,
        base_path=output_dir,
    )

    if verbose:
        print(f"\nMulti-model run complete: {run_id}")
        print(f"Directory: {full_output_dir}")

    return MultiModelEvalResult(
        run_id=run_id,
        run_path=full_output_dir,
        results=results,
        comparison=comparison,
    )

def run_walk_forward_evaluation(
    config: "EvalConfig",
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
    run_id: str | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> "MultiModelEvalResult":
    """Run walk-forward cross-validation evaluation.

    Uses WalkForwardSplit.generate_folds() to evaluate across all folds,
    then aggregates metrics. Each fold trains on a window and tests on the
    next step_size days.

    Args:
        config: Evaluation configuration (split.type must be "walk_forward")
        forecast_df: Forecast data
        truth_df: Truth data
        feature_df: Optional pre-built features
        run_id: Optional run ID
        output_dir: Optional output directory
        verbose: Whether to print progress

    Returns:
        MultiModelEvalResult with per-fold results and aggregated metrics
    """
    from tempdata.eval.config import generate_run_id
    from tempdata.eval.data import load_eval_data, _normalize_eval_dates, _ensure_features, _join_forecast_truth
    from tempdata.eval.metrics import (
        EvalMetrics,
        compute_accuracy_bands,
        compute_calibration_metrics,
        compute_forecast_metrics,
    )
    from tempdata.eval.models import create_forecaster
    from tempdata.eval.report import create_run_dir, write_model_artifacts, write_run_metadata, write_comparison_summary
    from tempdata.eval.slicing import compute_metrics_by_slice
    from tempdata.eval.splits import WalkForwardSplit

    if config.split.type != "walk_forward":
        raise ValueError("run_walk_forward_evaluation requires split.type='walk_forward'")

    if config.split.window_size is None or config.split.step_size is None:
        raise ValueError("walk_forward requires window_size and step_size")

    if run_id is None:
        run_id = generate_run_id()

    # Load and prepare data (same as load_eval_data but without splitting)
    if feature_df is not None:
        df = feature_df.copy()
    else:
        df = _join_forecast_truth(forecast_df, truth_df)

    df = _normalize_eval_dates(df)
    df = _ensure_features(df)
    df = df.sort_values(["station_id", "target_date_local"]).reset_index(drop=True)

    # Generate folds
    splitter = WalkForwardSplit(
        window_size=config.split.window_size,
        step_size=config.split.step_size,
    )
    folds = splitter.generate_folds(df)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"WALK-FORWARD EVALUATION: {run_id}")
        print(f"Folds: {len(folds)}")
        print(f"Window: {config.split.window_size}d, Step: {config.split.step_size}d")
        print(f"Model: {config.model.type}")
        print(f"{'=' * 60}\n")

    # Create run directory
    run_dir = create_run_dir(run_id, base_path=output_dir)
    write_run_metadata(
        run_id=run_id,
        run_name=f"Walk-Forward {config.model.type}",
        model_names=[f"fold_{i}" for i in range(len(folds))],
        base_path=output_dir,
    )

    fold_results = {}

    for i, (train_df, test_df) in enumerate(folds):
        if verbose:
            print(f"--- Fold {i+1}/{len(folds)}: train={len(train_df)}, test={len(test_df)} ---")

        # Create and fit model
        forecaster = create_forecaster(
            model_type=config.model.type,
            alpha=config.model.alpha,
            features=config.model.features,
        )
        forecaster.fit(train_df)

        # Generate predictions
        predictions_df = test_df.copy()
        predictions_df["y_pred_f"] = forecaster.predict_mu(test_df)
        predictions_df["y_true_f"] = test_df["tmax_actual_f"].values

        # Compute metrics
        forecast_metrics = compute_forecast_metrics(predictions_df)
        accuracy_bands = compute_accuracy_bands(predictions_df)

        calibration_metrics = None
        if "y_pred_sigma_f" in predictions_df.columns:
            try:
                calibration_metrics = compute_calibration_metrics(predictions_df)
            except ImportError:
                pass

        slices = compute_metrics_by_slice(predictions_df)

        metrics = EvalMetrics(
            forecast=forecast_metrics,
            calibration=calibration_metrics,
            accuracy_bands=accuracy_bands,
            slices=slices,
        )

        # Write fold artifacts
        fold_name = f"fold_{i}"
        artifacts = write_model_artifacts(
            model_name=fold_name,
            config=config,
            metrics=metrics,
            predictions_df=predictions_df,
            run_id=run_id,
            base_path=output_dir,
        )

        fold_results[fold_name] = EvalResult(
            run_id=run_id,
            config=config,
            predictions_df=predictions_df,
            metrics=metrics,
            artifacts=artifacts,
        )

        if verbose:
            fm = metrics.forecast
            ab = metrics.accuracy_bands
            print(f"  MAE={fm.mae:.2f} RMSE={fm.rmse:.2f} R2={fm.r2:.3f}")
            if ab:
                print(f"  +/-1F: {100*ab.within_1f:.1f}%  +/-2F: {100*ab.within_2f:.1f}%")

    # Aggregate fold metrics
    summary_data = {}
    for name, res in fold_results.items():
        summary_data[name] = {"metrics": res.metrics.to_dict()}

    comparison = write_comparison_summary(
        run_id=run_id,
        model_results=summary_data,
        base_path=output_dir,
    )

    # Print aggregate
    if verbose and fold_results:
        maes = [r.metrics.forecast.mae for r in fold_results.values()]
        rmses = [r.metrics.forecast.rmse for r in fold_results.values()]
        print(f"\n{'=' * 60}")
        print(f"AGGREGATE ({len(folds)} folds)")
        print(f"  MAE:  {sum(maes)/len(maes):.2f} (range: {min(maes):.2f} - {max(maes):.2f})")
        print(f"  RMSE: {sum(rmses)/len(rmses):.2f} (range: {min(rmses):.2f} - {max(rmses):.2f})")
        print(f"{'=' * 60}")

    return MultiModelEvalResult(
        run_id=run_id,
        run_path=run_dir,
        results=fold_results,
        comparison=comparison,
    )

