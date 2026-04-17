"""Generate reports for daily maximum temperature evaluation results.

This module provides higher-level reporting on top of the eval framework:
- Load and compare multiple runs
- Generate cross-run summary tables
- Trend analysis across model types
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_run_summary(run_dir: Path | str) -> dict[str, Any]:
    """Load a run's metadata and metrics into a summary dict.

    Args:
        run_dir: Path to a run directory (single-model or multi-model)

    Returns:
        Dictionary with run_id, run_name, models, and metrics
    """
    run_dir = Path(run_dir)
    summary: dict[str, Any] = {"run_dir": str(run_dir)}

    # Load meta
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        summary["run_id"] = meta.get("run_id")
        summary["run_name"] = meta.get("run_name")
        summary["timestamp"] = meta.get("timestamp_utc")
        summary["git_commit"] = meta.get("git_commit")

    # Single-model run
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        summary["type"] = "single_model"
        summary["metrics"] = json.loads(metrics_path.read_text())
        config_path = run_dir / "config.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            model_cfg = cfg.get("model", {})
            summary["model_type"] = model_cfg.get("type")
            summary["station_ids"] = cfg.get("station_ids")
            summary["features"] = model_cfg.get("features")
            summary["model_hyperparams"] = model_cfg.get("hyperparams", {})
        return summary

    # Multi-model run
    models_dir = run_dir / "models"
    if models_dir.exists() and any(models_dir.iterdir()):
        summary["type"] = "multi_model"
        summary["models"] = {}
        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_metrics_path = model_dir / "metrics.json"
            if model_metrics_path.exists():
                model_metrics = json.loads(model_metrics_path.read_text())
                summary["models"][model_dir.name] = model_metrics

        # Load comparison if available
        comparison_path = run_dir / "comparison.json"
        if comparison_path.exists():
            summary["comparison"] = json.loads(comparison_path.read_text())

        return summary

    summary["type"] = "unknown"
    return summary


def compare_runs(run_dirs: list[Path | str]) -> pd.DataFrame:
    """Compare multiple runs side-by-side.

    Args:
        run_dirs: List of run directory paths

    Returns:
        DataFrame with one row per model/run, columns for key metrics
    """
    rows = []
    for rd in run_dirs:
        s = load_run_summary(rd)
        run_id = s.get("run_id", Path(rd).name)

        if s["type"] == "single_model":
            fm = s.get("metrics", {}).get("forecast", {})
            ab = s.get("metrics", {}).get("accuracy_bands", {})
            features = s.get("features") or []
            rows.append({
                "run_id": run_id,
                "model": s.get("model_type", "?"),
                "feature_count": len(features),
                "features": ", ".join(features) if isinstance(features, list) else str(features),
                "mae": fm.get("mae"),
                "rmse": fm.get("rmse"),
                "bias": fm.get("bias"),
                "r2": fm.get("r2"),
                "n_samples": fm.get("n_samples"),
                "within_1f_pct": round(100 * ab.get("within_1f", 0), 1) if ab else None,
                "within_2f_pct": round(100 * ab.get("within_2f", 0), 1) if ab else None,
            })
        elif s["type"] == "multi_model":
            for model_name, metrics in s.get("models", {}).items():
                fm = metrics.get("forecast", {})
                ab = metrics.get("accuracy_bands", {})
                rows.append({
                    "run_id": run_id,
                    "model": model_name,
                    "mae": fm.get("mae"),
                    "rmse": fm.get("rmse"),
                    "bias": fm.get("bias"),
                    "r2": fm.get("r2"),
                    "n_samples": fm.get("n_samples"),
                    "within_1f_pct": round(100 * ab.get("within_1f", 0), 1) if ab else None,
                    "within_2f_pct": round(100 * ab.get("within_2f", 0), 1) if ab else None,
                })

    df = pd.DataFrame(rows)
    if not df.empty and "mae" in df.columns:
        df = df.sort_values("mae").reset_index(drop=True)
    return df


def list_all_runs(runs_dir: Path | str = "runs") -> pd.DataFrame:
    """List all runs with summary metrics.

    Args:
        runs_dir: Base directory containing run folders

    Returns:
        DataFrame with one row per run, sorted by timestamp
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return pd.DataFrame()

    summaries = []
    for child in sorted(runs_dir.iterdir()):
        if child.is_dir():
            s = load_run_summary(child)
            summaries.append(s)

    if not summaries:
        return pd.DataFrame()

    return pd.DataFrame(summaries)


def print_run_comparison(run_dirs: list[Path | str]) -> None:
    """Print a formatted comparison table of runs.

    Args:
        run_dirs: List of run directory paths
    """
    df = compare_runs(run_dirs)
    if df.empty:
        print("No runs to compare.")
        return

    print("\n" + "=" * 70)
    print("RUN COMPARISON")
    print("=" * 70)
    print(df.to_string(index=False))
    print()
