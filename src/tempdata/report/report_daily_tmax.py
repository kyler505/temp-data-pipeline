"""Generate reports for daily maximum temperature evaluation results.

This module provides higher-level reporting on top of the eval framework:
- Load and compare multiple runs
- Generate cross-run summary tables
- Trend analysis across model types
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
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


def format_live_multi_horizon_report(log_path: Path | str) -> str:
    """Format a multi-horizon live forecast report from predictions.jsonl.

    Returns a markdown-formatted string suitable for Discord/terminal output.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return "No predictions found."

    lines = log_path.read_text().strip().split("\n")
    preds = [json.loads(l) for l in lines if l.strip()]
    if not preds:
        return "No predictions found."

    # Group by horizon_days, deduplicate by most recent per (target_date, horizon)
    by_key = {}
    for p in preds:
        h = p.get("horizon_days", 1)
        key = (p["target_date"], h)
        existing = by_key.get(key)
        if existing is None or p.get("date", "") > existing.get("date", ""):
            by_key[key] = p

    today = date.today()
    unscored = [p for p in by_key.values() if p.get("actual_f") is None and p["target_date"] >= str(today)]
    unscored.sort(key=lambda p: p["target_date"])

    scored = [p for p in by_key.values() if p.get("actual_f") is not None]
    scored.sort(key=lambda p: (p["target_date"], p.get("horizon_days", 1)))

    model_type = next((p.get("model_type", "model") for p in by_key.values()), "model")
    model_label = model_type.title()

    # Build report
    lines_out = []

    # Today's forecasts
    if unscored:
        lines_out.append("## 🔮 Forecasts")
        lines_out.append("")
        lines_out.append("| Target | Horizon | Raw | {} | Correction |".format(model_label))
        lines_out.append("|--------|---------|-----|------|------------|")
        for p in unscored:
            h = p.get("horizon_days", 1)
            raw = p["raw_forecast_f"]
            model = p.get("model_prediction_f", p.get("ridge_prediction_f", raw))
            corr = model - raw
            lines_out.append("| {} | +{}d | {:.1f}°F | {:.1f}°F | {:+.1f}°F |".format(
                p["target_date"], h, raw, model, corr
            ))
        lines_out.append("")

    # Accuracy by horizon
    if scored:
        horizons = sorted(set(p.get("horizon_days", 1) for p in scored))
        lines_out.append("## 📊 Accuracy by Horizon")
        lines_out.append("")
        lines_out.append("| Horizon | N | Raw MAE | {} MAE | Raw ±1F | {} ±1F |".format(model_label, model_label))
        lines_out.append("|---------|---|---------|--------|---------|--------|")

        for h in horizons:
            h_preds = [p for p in scored if p.get("horizon_days", 1) == h]
            raw_errors = []
            model_errors = []
            for p in h_preds:
                raw_err = p.get("raw_error_f", p["raw_forecast_f"] - p["actual_f"])
                model_pred = p.get("model_prediction_f", p.get("ridge_prediction_f"))
                model_err = p.get("model_error_f", p.get("error_f", model_pred - p["actual_f"]))
                raw_errors.append(raw_err)
                model_errors.append(model_err)
            if raw_errors:
                raw_mae = np.mean(np.abs(raw_errors))
                model_mae = np.mean(np.abs(model_errors))
                raw_w1 = np.mean(np.abs(raw_errors) <= 1.0)
                model_w1 = np.mean(np.abs(model_errors) <= 1.0)
                lines_out.append("| +{}d | {} | {:.2f}°F | {:.2f}°F | {:.0%} | {:.0%} |".format(
                    h, len(h_preds), raw_mae, model_mae, raw_w1, model_w1
                ))

        # Recent scored details (last 10)
        recent = scored[-10:]
        lines_out.append("")
        lines_out.append("### Recent Scored Predictions")
        lines_out.append("")
        lines_out.append("| Date | H | Actual | Raw | {} | Raw Err | {} Err |".format(model_label, model_label))
        lines_out.append("|------|---|--------|-----|------|---------|--------|")
        for p in recent:
            h = p.get("horizon_days", 1)
            raw_err = p.get("raw_error_f", p["raw_forecast_f"] - p["actual_f"])
            model_pred = p.get("model_prediction_f", p.get("ridge_prediction_f"))
            model_err = p.get("model_error_f", p.get("error_f", model_pred - p["actual_f"]))
            lines_out.append("| {} | +{}d | {:.1f} | {:.1f} | {:.1f} | {:+.1f} | {:+.1f} |".format(
                p["target_date"], h, p["actual_f"], p["raw_forecast_f"],
                model_pred, raw_err, model_err
            ))

    # Config
    lines_out.append("")
    lines_out.append("---")
    horizons_used = sorted(set(p.get("horizon_days", 1) for p in preds))
    lines_out.append(f"**Model:** {model_type} | **Features:** {len(next(iter(by_key.values())).get('feature_set', []))} | **Horizons:** {', '.join(f'+{h}d' for h in horizons_used)}")

    return "\n".join(lines_out)
