"""Artifact generation and run management.

This module handles writing evaluation artifacts to disk:
- config.json: Frozen configuration
- meta.json: Run metadata (git hash, timestamp)
- predictions.parquet: Model predictions
- residuals.parquet: Residuals for analysis
- metrics.json: Evaluation metrics summary
- slices.json: Sliced metrics breakdown

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
    """Write all run artifacts to disk.

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


def list_runs(base_path: Path | None = None) -> list[dict[str, Any]]:
    """List all available runs.

    Args:
        base_path: Base directory for runs

    Returns:
        List of run info dictionaries with run_id and timestamp
    """
    if base_path is None:
        base_path = Path("runs")

    if not base_path.exists():
        return []

    runs = []
    for run_dir in sorted(base_path.iterdir(), reverse=True):
        if run_dir.is_dir():
            info = {"run_id": run_dir.name}

            # Try to load metadata
            meta_path = run_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    info["timestamp"] = meta.get("timestamp_utc")
                    info["run_name"] = meta.get("run_name")
                except json.JSONDecodeError:
                    pass

            runs.append(info)

    return runs
