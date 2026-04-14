"""Temperature evaluation framework for daily Tmax prediction quality.

This package intentionally avoids importing optional ML dependencies at import
time. Use `from tempdata.eval import EvalConfig` or import submodules directly.
Heavy symbols are loaded lazily via ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    # Config
    "EvalConfig": ("tempdata.eval.config", "EvalConfig"),
    "generate_run_id": ("tempdata.eval.config", "generate_run_id"),
    # Data
    "EvalDataset": ("tempdata.eval.data", "EvalDataset"),
    "load_eval_data": ("tempdata.eval.data", "load_eval_data"),
    "load_eval_inputs": ("tempdata.eval.data", "load_eval_inputs"),
    # Models
    "Forecaster": ("tempdata.eval.models", "Forecaster"),
    "RidgeForecaster": ("tempdata.eval.models", "RidgeForecaster"),
    "PassthroughForecaster": ("tempdata.eval.models", "PassthroughForecaster"),
    "create_forecaster": ("tempdata.eval.models", "create_forecaster"),
    # Uncertainty
    "UncertaintyModel": ("tempdata.eval.uncertainty", "UncertaintyModel"),
    "GlobalSigma": ("tempdata.eval.uncertainty", "GlobalSigma"),
    "BucketedSigma": ("tempdata.eval.uncertainty", "BucketedSigma"),
    "RollingSigma": ("tempdata.eval.uncertainty", "RollingSigma"),
    "create_uncertainty_model": ("tempdata.eval.uncertainty", "create_uncertainty_model"),
    # Splits
    "create_split": ("tempdata.eval.splits", "create_split"),
    "StaticSplit": ("tempdata.eval.splits", "StaticSplit"),
    "WalkForwardSplit": ("tempdata.eval.splits", "WalkForwardSplit"),
    # Metrics
    "ForecastMetrics": ("tempdata.eval.metrics", "ForecastMetrics"),
    "CalibrationMetrics": ("tempdata.eval.metrics", "CalibrationMetrics"),
    "compute_forecast_metrics": ("tempdata.eval.metrics", "compute_forecast_metrics"),
    "compute_calibration_metrics": ("tempdata.eval.metrics", "compute_calibration_metrics"),
    "compute_metrics_by_slice": ("tempdata.eval.slicing", "compute_metrics_by_slice"),
    # Report
    "create_run_dir": ("tempdata.eval.report", "create_run_dir"),
    "load_run": ("tempdata.eval.report", "load_run"),
    "load_multi_model_run": ("tempdata.eval.report", "load_multi_model_run"),
    "list_runs": ("tempdata.eval.report", "list_runs"),
    "write_all_artifacts": ("tempdata.eval.report", "write_all_artifacts"),
    # Runner
    "run_evaluation": ("tempdata.eval.runner", "run_evaluation"),
    "run_multi_model_evaluation": ("tempdata.eval.runner", "run_multi_model_evaluation"),
    "MultiModelEvalResult": ("tempdata.eval.runner", "MultiModelEvalResult"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
