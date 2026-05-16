#!/usr/bin/env python3
"""Post-run pipeline verification.

Validates predictions log, model artifacts, and data integrity.
Exits 0 on success, 1 on failure with error details.

Usage:
    python scripts/verify_pipeline.py [--station KNYC] [--log-dir runs/live]
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np


def verify_predictions(log_path: Path, station_id: str, horizons: list[int]) -> list[str]:
    """Validate predictions.jsonl contents."""
    errors = []
    warnings = []

    if not log_path.exists():
        errors.append(f"Predictions log missing: {log_path}")
        return errors

    lines = log_path.read_text().strip().split("\n")
    if not lines or not lines[0].strip():
        errors.append("Predictions log is empty")
        return errors

    predictions = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            pred = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i+1}: Invalid JSON: {e}")
            continue
        predictions.append(pred)

    today_str = date.today().isoformat()
    today_preds = [p for p in predictions if p.get("target_date") >= today_str and p.get("station_id") == station_id]

    if not today_preds:
        errors.append(f"No predictions found for today ({today_str}) for station {station_id}")
    else:
        found_horizons = {p.get("horizon_days") for p in today_preds}
        missing = set(horizons) - found_horizons
        if missing:
            errors.append(f"Missing horizons for today: {sorted(missing)}")

    required_fields = [
        "date", "target_date", "station_id", "raw_forecast_f",
        "model_prediction_f", "lead_hours", "horizon_days", "model_used",
    ]
    for i, pred in enumerate(predictions):
        for field in required_fields:
            if field not in pred or pred[field] is None:
                errors.append(f"Prediction {i+1}: Missing field '{field}'")
                break
        raw = pred.get("raw_forecast_f")
        if raw is not None and not (-20 <= raw <= 120):
            errors.append(f"Prediction {i+1}: raw_forecast_f {raw} out of bounds")
        model_pred = pred.get("model_prediction_f")
        if model_pred is not None and not (-20 <= model_pred <= 120):
            errors.append(f"Prediction {i+1}: model_prediction_f {model_pred} out of bounds")
        lead = pred.get("lead_hours")
        if lead is not None and lead < 0:
            errors.append(f"Prediction {i+1}: Negative lead_hours {lead}")

    seen = set()
    for pred in today_preds:
        key = (pred.get("target_date"), pred.get("horizon_days"), pred.get("date"))
        if key in seen:
            errors.append(f"Duplicate prediction in same run for {key[:2]}")
        seen.add(key)

    return errors + warnings


def verify_model_artifacts(log_dir: Path) -> list[str]:
    """Validate model versioning artifacts."""
    errors = []
    model_dir = log_dir.parent / "models"
    if not model_dir.exists():
        errors.append(f"Model directory missing: {model_dir}")
        return errors
    current_link = model_dir / "current.pkl"
    if not current_link.exists() and not current_link.is_symlink():
        errors.append("current.pkl symlink missing")
        return errors
    try:
        resolved = current_link.resolve()
    except Exception as e:
        errors.append(f"current.pkl broken symlink: {e}")
        return errors
    if not resolved.exists():
        errors.append(f"current.pkl points to missing file: {resolved}")
        return errors
    meta_path = resolved.with_suffix(".json")
    if not meta_path.exists():
        errors.append(f"Model metadata missing: {meta_path}")
    else:
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            required_meta = ["timestamp", "station_id", "model_type", "train_rows"]
            for field in required_meta:
                if field not in meta:
                    errors.append(f"Metadata missing field: {field}")
        except json.JSONDecodeError:
            errors.append(f"Invalid metadata JSON: {meta_path}")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Verify pipeline integrity")
    parser.add_argument("--station", default="KNYC")
    parser.add_argument("--log-dir", default="runs/live", type=Path)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 5, 7])
    args = parser.parse_args()
    log_path = args.log_dir / "predictions.jsonl"
    print(f"[verify] Checking predictions: {log_path}")
    issues = verify_predictions(log_path, args.station, args.horizons)
    print(f"[verify] Checking model artifacts")
    issues.extend(verify_model_artifacts(args.log_dir))
    if not issues:
        print("[verify] ✅ All checks passed")
        sys.exit(0)
    print(f"[verify] ❌ {len(issues)} issue(s) found:")
    for issue in issues:
        print(f"  - {issue}")
    sys.exit(1)


if __name__ == "__main__":
    main()