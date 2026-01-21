"""Artifact generation and run management.

This module handles writing evaluation artifacts to disk:
- config.json: Frozen configuration
- meta.json: Run metadata (git hash, timestamp)
- predictions.parquet: Model predictions
- residuals.parquet: Residuals for analysis
- metrics.json: Evaluation metrics summary
- slices.json: Sliced metrics breakdown

Supports both single-model and multi-model run structures:
- Single-model: artifacts at run_dir root
- Multi-model: artifacts in run_dir/models/{model_name}/

No trading artifacts (trades.parquet, daily_pnl.parquet, etc.)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tempdata.eval.config import EvalConfig
    from tempdata.eval.metrics import EvalMetrics


def create_run_dir(run_id: str, base_path: Path | None = None) -> Path:
    """Create a directory for run artifacts.

    Args:
        run_id: Unique run identifier
        base_path: Base directory (default: runs/)

    Returns:
        Path to the created run directory
    """
    if base_path is None:
        base_path = Path("runs")

    run_dir = base_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def write_all_artifacts(
    config: EvalConfig,
    metrics: EvalMetrics,
    predictions_df: pd.DataFrame,
    run_id: str,
    base_path: Path | None = None,
) -> dict[str, Path]:
    """Write all run artifacts to disk (single-model format).

    Args:
        config: Evaluation configuration
        metrics: Computed evaluation metrics
        predictions_df: DataFrame with predictions and actuals
        run_id: Unique run identifier
        base_path: Base directory for runs

    Returns:
        Dictionary of artifact name -> path
    """
    run_dir = create_run_dir(run_id, base_path)
    artifacts = {}

    # Write config
    config_path = run_dir / "config.json"
    config_path.write_text(config.to_json())
    artifacts["config"] = config_path

    # Write metadata
    meta = _create_metadata(config, run_id)
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    artifacts["meta"] = meta_path

    # Write predictions
    predictions_path = run_dir / "predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    artifacts["predictions"] = predictions_path

    # Write residuals
    residuals_df = _create_residuals_df(predictions_df)
    residuals_path = run_dir / "residuals.parquet"
    residuals_df.to_parquet(residuals_path, index=False)
    artifacts["residuals"] = residuals_path

    # Write metrics
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2))
    artifacts["metrics"] = metrics_path

    # Write slices
    if metrics.slices:
        slices_path = run_dir / "slices.json"
        slices_path.write_text(json.dumps(metrics.slices, indent=2))
        artifacts["slices"] = slices_path

    return artifacts


def write_model_artifacts(
    model_name: str,
    config: EvalConfig,
    metrics: EvalMetrics,
    predictions_df: pd.DataFrame,
    run_id: str,
    base_path: Path | None = None,
) -> dict[str, Path]:
    """Write model-specific artifacts to a subdirectory within a run.

    For multi-model runs, each model gets its own subdirectory under
    run_dir/models/{model_name}/

    Args:
        model_name: Name of the model (e.g., 'ridge', 'persistence')
        config: Evaluation configuration
        metrics: Computed evaluation metrics
        predictions_df: DataFrame with predictions and actuals
        run_id: Unique run identifier
        base_path: Base directory for runs

    Returns:
        Dictionary of artifact name -> path
    """
    run_dir = create_run_dir(run_id, base_path)
    model_dir = run_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    # Write config
    config_path = model_dir / "config.json"
    config_path.write_text(config.to_json())
    artifacts["config"] = config_path

    # Write predictions
    predictions_path = model_dir / "predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    artifacts["predictions"] = predictions_path

    # Write residuals
    residuals_df = _create_residuals_df(predictions_df)
    residuals_path = model_dir / "residuals.parquet"
    residuals_df.to_parquet(residuals_path, index=False)
    artifacts["residuals"] = residuals_path

    # Write metrics
    metrics_path = model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2))
    artifacts["metrics"] = metrics_path

    # Write slices
    if metrics.slices:
        slices_path = model_dir / "slices.json"
        slices_path.write_text(json.dumps(metrics.slices, indent=2))
        artifacts["slices"] = slices_path

    return artifacts


def write_run_metadata(
    run_id: str,
    run_name: str,
    model_names: list[str],
    base_path: Path | None = None,
) -> Path:
    """Write run-level metadata for a multi-model run.

    Args:
        run_id: Unique run identifier
        run_name: Human-readable run name
        model_names: List of model names in this run
        base_path: Base directory for runs

    Returns:
        Path to the metadata file
    """
    import sys
    import subprocess

    run_dir = create_run_dir(run_id, base_path)

    # Get git commit
    git_commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()[:8]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    meta = {
        "run_id": run_id,
        "run_name": run_name,
        "git_commit": git_commit,
        "python_version": sys.version,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "is_multi_model": True,
        "model_names": model_names,
    }

    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return meta_path


def write_comparison_summary(
    run_id: str,
    model_results: dict[str, dict[str, Any]],
    base_path: Path | None = None,
) -> Path:
    """Write a comparison summary aggregating metrics from all models.

    Args:
        run_id: Unique run identifier
        model_results: Dict mapping model_name -> metrics dict
        base_path: Base directory for runs

    Returns:
        Path to the comparison summary file
    """
    run_dir = create_run_dir(run_id, base_path)

    # Build comparison summary
    comparison = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "models": {},
        "ranking": {},
    }

    # Extract key metrics for comparison
    mae_values = {}
    rmse_values = {}
    bias_values = {}

    for model_name, result in model_results.items():
        metrics = result.get("metrics", {})
        forecast = metrics.get("forecast", {})

        comparison["models"][model_name] = {
            "mae": forecast.get("mae"),
            "rmse": forecast.get("rmse"),
            "bias": forecast.get("bias"),
            "r2": forecast.get("r2"),
            "n_samples": forecast.get("n_samples"),
        }

        if forecast.get("mae") is not None:
            mae_values[model_name] = forecast["mae"]
        if forecast.get("rmse") is not None:
            rmse_values[model_name] = forecast["rmse"]
        if forecast.get("bias") is not None:
            bias_values[model_name] = abs(forecast["bias"])

    # Create rankings (lower is better for MAE, RMSE, |bias|)
    if mae_values:
        comparison["ranking"]["by_mae"] = sorted(
            mae_values.keys(), key=lambda x: mae_values[x]
        )
    if rmse_values:
        comparison["ranking"]["by_rmse"] = sorted(
            rmse_values.keys(), key=lambda x: rmse_values[x]
        )
    if bias_values:
        comparison["ranking"]["by_abs_bias"] = sorted(
            bias_values.keys(), key=lambda x: bias_values[x]
        )

    comparison_path = run_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))

    return comparison_path


def _create_metadata(config: EvalConfig, run_id: str) -> dict[str, Any]:
    """Create run metadata dictionary."""
    import sys
    import subprocess

    # Get git commit
    git_commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()[:8]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return {
        "run_id": run_id,
        "run_name": config.run_name,
        "git_commit": git_commit,
        "python_version": sys.version,
        "timestamp_utc": datetime.utcnow().isoformat(),
    }


def _create_residuals_df(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Create residuals DataFrame for analysis.

    Args:
        predictions_df: DataFrame with y_pred_f and y_true_f

    Returns:
        DataFrame with residual analysis columns
    """
    df = predictions_df.copy()

    # Core residual columns
    df["residual_f"] = df["y_pred_f"] - df["y_true_f"]
    df["abs_error_f"] = df["residual_f"].abs()
    df["sq_error_f"] = df["residual_f"] ** 2

    # Keep key columns
    keep_cols = [
        "station_id", "target_date_local", "issue_time_utc",
        "y_pred_f", "y_true_f", "residual_f", "abs_error_f", "sq_error_f",
    ]

    # Add optional columns if present
    optional = ["lead_hours", "month", "doy", "y_pred_sigma_f"]
    for col in optional:
        if col in df.columns:
            keep_cols.append(col)

    return df[[c for c in keep_cols if c in df.columns]]


def load_run(run_id: str, base_path: Path | None = None) -> dict[str, Any]:
    """Load artifacts from a completed run.

    Handles both single-model and multi-model run structures.
    For multi-model runs, returns the first model's data for backward
    compatibility. Use load_multi_model_run() for full multi-model access.

    Args:
        run_id: Run identifier
        base_path: Base directory for runs

    Returns:
        Dictionary with config, metrics, and DataFrames
    """
    if base_path is None:
        base_path = Path("runs")

    run_dir = base_path / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")

    result = {"run_id": run_id, "run_dir": run_dir}

    # Check if this is a multi-model run
    models_dir = run_dir / "models"
    if models_dir.exists() and models_dir.is_dir():
        # Multi-model run: load metadata and first model for compatibility
        result["is_multi_model"] = True

        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            result["meta"] = json.loads(meta_path.read_text())

        # Load first model's data for backward compatibility
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        if model_dirs:
            first_model = sorted(model_dirs)[0]
            result["model_name"] = first_model.name
            result.update(_load_model_dir(first_model))

        return result

    # Single-model run: load from root
    result["is_multi_model"] = False

    # Load config
    config_path = run_dir / "config.json"
    if config_path.exists():
        from tempdata.eval.config import EvalConfig
        result["config"] = EvalConfig.load(config_path)

    # Load metadata
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        result["meta"] = json.loads(meta_path.read_text())

    # Load metrics
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        result["metrics"] = json.loads(metrics_path.read_text())

    # Load slices
    slices_path = run_dir / "slices.json"
    if slices_path.exists():
        result["slices"] = json.loads(slices_path.read_text())

    # Load DataFrames
    predictions_path = run_dir / "predictions.parquet"
    if predictions_path.exists():
        result["predictions_df"] = pd.read_parquet(predictions_path)

    residuals_path = run_dir / "residuals.parquet"
    if residuals_path.exists():
        result["residuals_df"] = pd.read_parquet(residuals_path)

    return result


def _load_model_dir(model_dir: Path) -> dict[str, Any]:
    """Load artifacts from a model subdirectory.

    Args:
        model_dir: Path to model directory

    Returns:
        Dictionary with config, metrics, and DataFrames
    """
    result = {}

    # Load config
    config_path = model_dir / "config.json"
    if config_path.exists():
        from tempdata.eval.config import EvalConfig
        result["config"] = EvalConfig.load(config_path)

    # Load metrics
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        result["metrics"] = json.loads(metrics_path.read_text())

    # Load slices
    slices_path = model_dir / "slices.json"
    if slices_path.exists():
        result["slices"] = json.loads(slices_path.read_text())

    # Load DataFrames
    predictions_path = model_dir / "predictions.parquet"
    if predictions_path.exists():
        result["predictions_df"] = pd.read_parquet(predictions_path)

    residuals_path = model_dir / "residuals.parquet"
    if residuals_path.exists():
        result["residuals_df"] = pd.read_parquet(residuals_path)

    return result


def load_multi_model_run(
    run_id: str, base_path: Path | None = None
) -> dict[str, Any]:
    """Load all models from a multi-model run.

    Args:
        run_id: Run identifier
        base_path: Base directory for runs

    Returns:
        Dictionary with:
        - run_id: str
        - run_dir: Path
        - meta: Run-level metadata
        - comparison: Comparison summary if available
        - models: Dict mapping model_name -> model data
    """
    if base_path is None:
        base_path = Path("runs")

    run_dir = base_path / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")

    result = {
        "run_id": run_id,
        "run_dir": run_dir,
        "models": {},
    }

    # Load run-level metadata
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        result["meta"] = json.loads(meta_path.read_text())

    # Load comparison summary
    comparison_path = run_dir / "comparison.json"
    if comparison_path.exists():
        result["comparison"] = json.loads(comparison_path.read_text())

    # Check for multi-model structure
    models_dir = run_dir / "models"
    if models_dir.exists() and models_dir.is_dir():
        result["is_multi_model"] = True
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                result["models"][model_name] = _load_model_dir(model_dir)
    else:
        # Single-model run: wrap in models dict
        result["is_multi_model"] = False
        model_name = result.get("meta", {}).get("run_name", "default")
        result["models"][model_name] = load_run(run_id, base_path)

    return result


def list_runs(base_path: Path | None = None) -> list[dict[str, Any]]:
    """List all available runs.

    Args:
        base_path: Base directory for runs

    Returns:
        List of run info dictionaries with run_id, timestamp,
        and is_multi_model flag
    """
    if base_path is None:
        base_path = Path("runs")

    if not base_path.exists():
        return []

    runs = []
    for run_dir in sorted(base_path.iterdir(), reverse=True):
        if run_dir.is_dir():
            info = {"run_id": run_dir.name}

            # Check if multi-model run
            models_dir = run_dir / "models"
            info["is_multi_model"] = models_dir.exists() and models_dir.is_dir()

            # Try to load metadata
            meta_path = run_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    info["timestamp"] = meta.get("timestamp_utc")
                    info["run_name"] = meta.get("run_name")
                    if info["is_multi_model"]:
                        info["model_names"] = meta.get("model_names", [])
                except json.JSONDecodeError:
                    pass

            runs.append(info)

    return runs
