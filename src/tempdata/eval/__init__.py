"""Temperature evaluation framework for daily Tmax prediction quality.

This module provides a pure temperature evaluation pipeline:
    features → model → predictions → temperature metrics + calibration diagnostics

Key components:
    - EvalConfig: Configuration for evaluation runs
    - Forecaster: Model interface for point predictions
    - UncertaintyModel: Interface for calibrated uncertainty
    - ForecastMetrics: MAE, RMSE, bias, calibration metrics

Example usage:
    from tempdata.eval import (
        EvalConfig,
        load_eval_data,
        run_evaluation,
        create_forecaster,
        create_uncertainty_model,
    )

    config = EvalConfig.load("configs/eval_klga_v1.json")
    results = run_evaluation(config)
"""

from tempdata.eval.config import EvalConfig, generate_run_id
from tempdata.eval.data import EvalDataset, load_eval_data
from tempdata.eval.metrics import (
    CalibrationMetrics,
    ForecastMetrics,
    compute_forecast_metrics,
    compute_calibration_metrics,
)
from tempdata.eval.models import (
    Forecaster,
    PassthroughForecaster,
    RidgeForecaster,
    create_forecaster,
)
from tempdata.eval.report import (
    create_run_dir,
    write_all_artifacts,
    load_run,
    list_runs,
)
from tempdata.eval.runner import run_evaluation
from tempdata.eval.slicing import compute_metrics_by_slice
from tempdata.eval.splits import create_split, StaticSplit, WalkForwardSplit
from tempdata.eval.uncertainty import (
    UncertaintyModel,
    GlobalSigma,
    BucketedSigma,
    RollingSigma,
    create_uncertainty_model,
)

__all__ = [
    # Config
    "EvalConfig",
    "generate_run_id",
    # Data
    "EvalDataset",
    "load_eval_data",
    # Models
    "Forecaster",
    "RidgeForecaster",
    "PassthroughForecaster",
    "create_forecaster",
    # Uncertainty
    "UncertaintyModel",
    "GlobalSigma",
    "BucketedSigma",
    "RollingSigma",
    "create_uncertainty_model",
    # Splits
    "create_split",
    "StaticSplit",
    "WalkForwardSplit",
    # Metrics
    "ForecastMetrics",
    "CalibrationMetrics",
    "compute_forecast_metrics",
    "compute_calibration_metrics",
    "compute_metrics_by_slice",
    # Report
    "create_run_dir",
    "write_all_artifacts",
    "load_run",
    "list_runs",
    # Runner
    "run_evaluation",
]
