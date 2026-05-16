#!/usr/bin/env python3
"""Live daily temperature forecast with auto-retrain.

Fetches today's Open-Meteo forecast, backfills yesterday's actuals into
training data, trains ridge on updated history, predicts today's tmax,
and logs the result.

Usage:
    python scripts/live_forecast.py [--station KNYC] [--data-dir data] [--log-dir runs/live]
    python scripts/live_forecast.py --report     # Show accuracy history

Station defaults to KNYC (Central Park). Set TEMP_DATA_STATION env var to override.
"""
import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from zoneinfo import ZoneInfo

# Retry on transient errors: 5xx, rate limits, network blips
def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, requests.exceptions.RequestException):
        if exc.response is not None:
            status = exc.response.status_code
            if status == 429 or (500 <= status < 600):
                return True
    return False

def _retry(fn):
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )(fn)

sys.path.insert(0, "src")

from tempdata.eval.models import create_forecaster
from tempdata.eval.metrics import compute_forecast_metrics, compute_accuracy_bands


def _fetch_with_retry(url: str, params: dict, timeout: int = 30):
    """HTTP GET with retry for transient Open-Meteo errors."""
    import requests as _requests

    def _do_get():
        resp = _requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp

    # Apply tenacity.retry by calling the wrapped function
    wrapped = _retry(_do_get)
    return wrapped()


STATIONS = {
    "KLGA": {"lat": 40.7769, "lon": -73.8740, "tz": "America/New_York", "name": "LaGuardia"},
    "KNYC": {"lat": 40.7789, "lon": -73.9689, "tz": "America/New_York", "name": "Central Park"},
}

LIVE_MODEL_TYPE = "stacked"
LIVE_HORIZONS = [1, 2, 3, 5, 7]  # Forecast horizons in days
LIVE_FEATURES = [
    "tmax_pred_f",
    "lead_hours",
    "month",
    "bias_7d",
    "bias_14d",
    "bias_30d",
    "rmse_14d",
    "rmse_30d",
    "sigma_lead",
]
LIVE_FEATURES_BY_HORIZON = {
    'short': ['tmax_pred_f', 'lead_hours', 'month', 'bias_7d', 'bias_14d', 'rmse_14d'],
    'medium': ['tmax_pred_f', 'month', 'bias_14d', 'bias_30d', 'rmse_30d', 'sigma_lead'],
    'long': ['tmax_pred_f', 'month', 'bias_30d', 'rmse_30d', 'sigma_lead'],
}
LIVE_STACKED_HYPERPARAMS = {
    "stacking_splits": None,
    "xgboost": {"learning_rate": 0.05, "max_depth": 3},
}


def fetch_live_forecast(station_id: str) -> pd.DataFrame:
    """Fetch today's and tomorrow's forecast from Open-Meteo live API."""
    meta = STATIONS.get(station_id)
    if not meta:
        raise ValueError(f"Unknown station: {station_id}")

    tz = ZoneInfo(meta["tz"])
    params = {
        "latitude": meta["lat"],
        "longitude": meta["lon"],
        "daily": "temperature_2m_max",
        "timezone": meta["tz"],
        "forecast_days": 8,
    }

    resp = _fetch_with_retry("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    data = resp.json()

    daily = data["daily"]
    issue_time = datetime.now(timezone.utc)
    rows = []

    for i, date_str in enumerate(daily["time"]):
        tmax_c = daily["temperature_2m_max"][i]
        if tmax_c is None:
            continue
        tmax_f = tmax_c * 9 / 5 + 32
        target_date = pd.Timestamp(date_str)
        lead_hours = max(0, int(
            (target_date.tz_localize(meta["tz"]).tz_convert("UTC") - issue_time).total_seconds() // 3600
        ))
        rows.append({
            "station_id": station_id,
            "issue_time_utc": issue_time.isoformat(),
            "target_date_local": target_date,
            "tmax_pred_f": tmax_f,
            "lead_hours": lead_hours,
            "forecast_source": "openmeteo",
        })

    return pd.DataFrame(rows)


def fetch_actual_tmax(station_id: str, target_date: date) -> float | None:
    """Fetch yesterday's actual tmax from Open-Meteo historical API."""
    meta = STATIONS.get(station_id)
    if not meta:
        return None

    date_str = target_date.isoformat()
    params = {
        "latitude": meta["lat"],
        "longitude": meta["lon"],
        "daily": "temperature_2m_max",
        "timezone": meta["tz"],
        "start_date": date_str,
        "end_date": date_str,
    }

    try:
        resp = _fetch_with_retry(
            "https://historical-forecast-api.open-meteo.com/v1/forecast",
            params=params,
            timeout=30,
        )
        data = resp.json()
        tmax_c = data["daily"]["temperature_2m_max"][0]
        if tmax_c is not None:
            return tmax_c * 9 / 5 + 32
    except Exception:
        pass
    return None


def load_and_prepare_training_data(
    data_dir: Path,
    station_id: str,
    days_back: int = 365,
) -> pd.DataFrame:
    """Load historical feature table and prepare for training."""
    from tempdata.eval.data import _load_feature_input

    feature_df = _load_feature_input(data_dir, station_id, None)
    if feature_df is None:
        raise FileNotFoundError(f"No historical feature data found for {station_id}")

    feature_df = feature_df.copy()
    feature_df["target_date_local"] = pd.to_datetime(feature_df["target_date_local"]).dt.date

    cutoff = date.today() - pd.Timedelta(days=days_back)
    feature_df = feature_df[feature_df["target_date_local"] >= cutoff].reset_index(drop=True)

    defaults = {
        "tmax_pred_f": 0.0,
        "lead_hours": 0,
        "month": 0,
        "bias_7d": 0.0,
        "bias_14d": 0.0,
        "bias_30d": 0.0,
        "rmse_14d": 1.0,
        "rmse_30d": 1.0,
        "sigma_lead": 1.0,
    }
    for col, default in defaults.items():
        if col not in feature_df.columns:
            feature_df[col] = default

    if "tmax_actual_f" not in feature_df.columns:
        raise ValueError("Historical data missing tmax_actual_f column")

    return feature_df


def backfill_actuals(
    log_path: Path,
    train_df: pd.DataFrame,
    station_id: str,
    feature_path: Path,
) -> tuple[pd.DataFrame, int]:
    """Check predictions log for old entries, fetch actuals, append to training data.

    Returns updated train_df and count of backfilled entries.
    """
    if not log_path.exists():
        return train_df, 0

    lines = log_path.read_text().strip().split("\n")
    predictions = [json.loads(l) for l in lines if l.strip()]

    today = date.today()
    backfilled = 0
    new_rows = []

    # Find predictions from past days that don't have actuals yet
    existing_dates = set(train_df["target_date_local"].tolist())

    for pred in predictions:
        pred_date = date.fromisoformat(pred["target_date"])
        # Only backfill past dates
        if pred_date >= today:
            continue
        # Skip if already in training data
        if pred_date in existing_dates:
            continue
        # Skip if already has actual recorded
        if pred.get("actual_f") is not None:
            continue

        actual = fetch_actual_tmax(station_id, pred_date)
        if actual is None:
            continue

        model_pred = pred.get("model_prediction_f", pred.get("ridge_prediction_f"))

        # Update the prediction log
        pred["actual_f"] = round(actual, 1)
        pred["model_error_f"] = round(model_pred - actual, 1)
        pred["error_f"] = pred["model_error_f"]
        pred["raw_error_f"] = round(pred["raw_forecast_f"] - actual, 1)
        backfilled += 1

        # Build a training row
        target_ts = pd.Timestamp(pred_date)
        doy = target_ts.dayofyear
        new_rows.append({
            "station_id": station_id,
            "target_date_local": pred_date,
            "tmax_pred_f": pred["raw_forecast_f"],
            "lead_hours": pred.get("lead_hours", 24),
            "tmax_actual_f": actual,
            "month": pred_date.month,
            "bias_7d": 0.0,
            "bias_14d": 0.0,
            "bias_30d": 0.0,
            "rmse_14d": 1.0,
            "rmse_30d": 1.0,
            "sigma_lead": 1.0,
        })

    # Write back updated log
    if backfilled > 0:
        with open(log_path, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        print(f"[live] Backfilled {backfilled} actual(s)")

        # Append new rows to feature parquet
        new_df = pd.DataFrame(new_rows)
        train_df = pd.concat([train_df, new_df], ignore_index=True)
        train_df = train_df.sort_values("target_date_local").reset_index(drop=True)

        # Persist to parquet
        try:
            existing = pd.read_parquet(feature_path)
            new_df = new_df.copy()
            new_df["target_date_local"] = pd.to_datetime(new_df["target_date_local"])
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_parquet(feature_path, index=False)
            print(f"[live] Appended {backfilled} row(s) to {feature_path}")
        except Exception as e:
            print(f"[live] WARNING: could not persist to parquet: {e}")

    return train_df, backfilled


def build_live_features(forecast_row: pd.Series, historical_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature row for a live forecast using historical context."""
    target_date = forecast_row["target_date_local"]
    if isinstance(target_date, str):
        target_date = pd.Timestamp(target_date)

    hist = historical_df.sort_values("target_date_local").copy()
    if "y_pred_f" not in hist.columns:
        hist["y_pred_f"] = hist["tmax_pred_f"]
    hist["error"] = hist["tmax_actual_f"] - hist["y_pred_f"]

    bias_7d = hist["error"].tail(7).mean()
    bias_14d = hist["error"].tail(14).mean()
    bias_30d = hist["error"].tail(30).mean()

    rmse_14d = hist["rmse_14d"].iloc[-1] if "rmse_14d" in hist.columns else 1.0
    rmse_30d = hist["rmse_30d"].iloc[-1] if "rmse_30d" in hist.columns else rmse_14d
    sigma_lead = hist["sigma_lead"].iloc[-1] if "sigma_lead" in hist.columns else 1.0

    features = {
        "station_id": forecast_row["station_id"],
        "target_date_local": target_date,
        "tmax_pred_f": forecast_row["tmax_pred_f"],
        "lead_hours": forecast_row["lead_hours"],
        "month": target_date.month,
        "bias_7d": bias_7d,
        "bias_14d": bias_14d,
        "bias_30d": bias_30d,
        "rmse_14d": rmse_14d,
        "rmse_30d": rmse_30d,
        "sigma_lead": sigma_lead,
    }
    return pd.DataFrame([features])


def _get_model_dir(log_dir: Path) -> Path:
    """Directory where model versions are stored."""
    return log_dir.parent / "models"


def _scored_predictions_by_horizon(log_path: Path) -> dict[int, int]:
    """Count scored predictions per horizon from the log."""
    if not log_path.exists():
        return {}
    lines = log_path.read_text().strip().split("\n")
    counts: dict[int, int] = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            pred = json.loads(line)
        except json.JSONDecodeError:
            continue
        if pred.get("actual_f") is None:
            continue
        h = pred.get("horizon_days", 1)
        counts[h] = counts.get(h, 0) + 1
    return counts


def save_model_version(
    model,
    train_rows: int,
    station_id: str,
    model_type: str,
    alpha: float,
    horizons: list[int],
    recent_mae: float,
    log_dir: Path,
) -> Path | None:
    """Save model artifact with metadata. Only saves if train_rows changed by >= 5."""
    model_dir = _get_model_dir(log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check previous version metadata
    current_link = model_dir / "current.pkl"
    prev_meta = None
    if current_link.exists() or current_link.is_symlink():
        try:
            # Resolve symlink to get the target file, then look for its .json
            resolved = current_link.resolve()
            prev_meta_path = resolved.with_suffix(".json")
            if prev_meta_path.exists():
                with open(prev_meta_path) as f:
                    prev_meta = json.load(f)
        except Exception:
            pass

    if prev_meta and prev_meta.get("train_rows", 0) > train_rows + 5:
        return None  # Don't overwrite a better model with less data

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"{timestamp}.pkl"
    meta_path = model_dir / f"{timestamp}.json"

    with open(model_path, "wb") as f:
        cloudpickle.dump(model, f)

    meta = {
        "timestamp": timestamp,
        "station_id": station_id,
        "model_type": model_type,
        "train_rows": train_rows,
        "features": LIVE_FEATURES,
        "horizons": horizons,
        "recent_mae": round(recent_mae, 3),
        "alpha": alpha,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Update symlink (remove old if exists)
    if current_link.exists() or current_link.is_symlink():
        current_link.unlink()
    current_link.symlink_to(model_path.name)

    print(f"[live] Saved model version: {model_path.name} ({train_rows} rows, MAE={recent_mae:.2f})")
    return model_path


def load_model_version(log_dir: Path):
    """Load current model version if available. Returns None on failure."""
    model_dir = _get_model_dir(log_dir)
    current_link = model_dir / "current.pkl"
    if not current_link.exists() and not current_link.is_symlink():
        return None
    try:
        resolved = current_link.resolve()
        if not resolved.exists():
            return None
        with open(resolved, "rb") as f:
            return cloudpickle.load(f)
    except Exception as e:
        print(f"[live] WARNING: Failed to load model version: {e}")
        return None


def validate_prediction(result: dict) -> tuple[bool, str]:
    """Validate a prediction dict before writing to log."""
    required = ["target_date", "raw_forecast_f", "model_prediction_f", "lead_hours", "horizon_days"]
    for field in required:
        if field not in result or result[field] is None:
            return False, f"Missing required field: {field}"
        if isinstance(result[field], float) and (np.isnan(result[field]) or np.isinf(result[field])):
            return False, f"Invalid numeric value for {field}"
    if result["lead_hours"] < 0:
        return False, f"Negative lead_hours: {result['lead_hours']}"
    if not (-20 <= result["raw_forecast_f"] <= 120):
        return False, f"Raw forecast out of bounds: {result['raw_forecast_f']}"
    if not (-20 <= result["model_prediction_f"] <= 120):
        return False, f"Model prediction out of bounds: {result['model_prediction_f']}"
    return True, ""


def sanity_check_forecast(forecast_df: pd.DataFrame) -> list[str]:
    """Run sanity checks on fetched forecast data. Returns list of warnings."""
    warnings = []
    for _, row in forecast_df.iterrows():
        raw = row["tmax_pred_f"]
        if pd.isna(raw) or np.isinf(raw):
            warnings.append(f"NaN/Inf raw forecast for {row.get('target_date_local', 'unknown')}")
        elif not (-20 <= raw <= 120):
            warnings.append(f"Raw forecast {raw:.1f}°F out of bounds for {row.get('target_date_local', 'unknown')}")
        lead = row.get("lead_hours")
        if pd.isna(lead) or lead < 0:
            warnings.append(f"Invalid lead_hours {lead} for {row.get('target_date_local', 'unknown')}")
    return warnings


def print_accuracy_report(log_path: Path) -> None:
    """Print accuracy history from the predictions log, broken down by horizon."""
    if not log_path.exists():
        print("No prediction history found.")
        return

    lines = log_path.read_text().strip().split("\n")
    preds = [json.loads(l) for l in lines if l.strip()]

    scored = [p for p in preds if p.get("actual_f") is not None]
    if not scored:
        print("No scored predictions yet (actuals not available).")
        print(f"Total predictions: {len(preds)}")
        return

    # Deduplicate: keep the most recent prediction per (target_date, horizon_days)
    by_key = {}
    for p in scored:
        h = p.get("horizon_days", 1)
        key = (p["target_date"], h)
        existing = by_key.get(key)
        if existing is None or p.get("date", "") > existing.get("date", ""):
            by_key[key] = p
    scored = sorted(by_key.values(), key=lambda p: (p["target_date"], p.get("horizon_days", 1)))

    model_name = next(
        (p.get("model_type") for p in scored if p.get("model_type")),
        None,
    )
    if model_name is None:
        if any(p.get("model_prediction_f") is not None for p in scored):
            model_name = "stacked"
        elif any(p.get("ridge_prediction_f") is not None for p in scored):
            model_name = "ridge"
        else:
            model_name = "model"
    # Use neutral label for column headers; model_used field provides per-row detail
    model_label = "Prediction"

    def _compute_metrics(pred_list):
        raw_errors = []
        model_errors = []
        for p in pred_list:
            raw_err = p.get("raw_error_f", p["raw_forecast_f"] - p["actual_f"])
            model_pred = p.get("model_prediction_f", p.get("ridge_prediction_f"))
            model_err = p.get("model_error_f", p.get("error_f", model_pred - p["actual_f"]))
            raw_errors.append(raw_err)
            model_errors.append(model_err)
        if not raw_errors:
            return None
        return {
            "n": len(raw_errors),
            "raw_mae": np.mean(np.abs(raw_errors)),
            "model_mae": np.mean(np.abs(model_errors)),
            "raw_rmse": np.sqrt(np.mean(np.array(raw_errors) ** 2)),
            "model_rmse": np.sqrt(np.mean(np.array(model_errors) ** 2)),
            "raw_within_1": np.mean(np.abs(raw_errors) <= 1.0),
            "model_within_1": np.mean(np.abs(model_errors) <= 1.0),
            "raw_within_2": np.mean(np.abs(raw_errors) <= 2.0),
            "model_within_2": np.mean(np.abs(model_errors) <= 2.0),
        }

    def _print_metrics_table(label, metrics):
        print(f"\n{'=' * 65}")
        print(f"{label}")
        print("=" * 65)
        print(f"{'Date':<12} {'H':>3} {'Actual':>6} {'Raw':>6} {model_label:>6} {'RawErr':>7} {model_label+'Err':>8}")
        print("-" * 65)

    pending_by_key = {}
    for p in preds:
        if p.get("actual_f") is not None:
            continue
        h = p.get("horizon_days", 1)
        key = (p["target_date"], h)
        existing = pending_by_key.get(key)
        if existing is None or p.get("date", "") > existing.get("date", ""):
            pending_by_key[key] = p

    pending_counts = {}
    for p in pending_by_key.values():
        h = p.get("horizon_days", 1)
        pending_counts[h] = pending_counts.get(h, 0) + 1

    # Group by horizon
    horizons = sorted(set(p.get("horizon_days", 1) for p in scored))

    # Overall report
    _print_metrics_table("LIVE ACCURACY TRACKING (ALL HORIZONS)", scored)
    for p in scored:
        h = p.get("horizon_days", 1)
        raw_err = p.get("raw_error_f", p["raw_forecast_f"] - p["actual_f"])
        model_pred = p.get("model_prediction_f", p.get("ridge_prediction_f"))
        model_err = p.get("model_error_f", p.get("error_f", model_pred - p["actual_f"]))
        print(f"{p['target_date']:<12} {h:>2}d {p['actual_f']:>6.1f} {p['raw_forecast_f']:>6.1f} "
              f"{model_pred:>6.1f} {raw_err:>+7.1f} {model_err:>+8.1f}")

    overall = _compute_metrics(scored)
    if overall:
        print("-" * 65)
        print(f"{'MAE':<16} {overall['raw_mae']:>6.2f} {overall['model_mae']:>6.2f}")
        print(f"{'RMSE':<16} {overall['raw_rmse']:>6.2f} {overall['model_rmse']:>6.2f}")
        print(f"{'±1F':<16} {overall['raw_within_1']:>5.0%} {overall['model_within_1']:>5.0%}")
        print(f"{'±2F':<16} {overall['raw_within_2']:>5.0%} {overall['model_within_2']:>5.0%}")

    # Per-horizon breakdown
    if len(horizons) > 1:
        print(f"\n{'=' * 65}")
        print("PER-HORIZON BREAKDOWN")
        print("=" * 65)
        print(f"{'Horizon':<12} {'N':>4} {'Raw MAE':>8} {model_label+' MAE':>10} {'Raw ±1F':>8} {model_label+' ±1F':>10} {'Improvement':>12}")
        print("-" * 65)
        for h in horizons:
            h_preds = [p for p in scored if p.get("horizon_days", 1) == h]
            m = _compute_metrics(h_preds)
            if m:
                imp = m["raw_mae"] - m["model_mae"]
                print(f"h={h}d         {m['n']:>4} {m['raw_mae']:>7.2f}° {m['model_mae']:>9.2f}° "
                      f"{m['raw_within_1']:>7.0%} {m['model_within_1']:>9.0%} {imp:>+11.2f}°")

    print(f"\nPredictions: {len(scored)} scored / {len(preds)} total")
    print(f"Horizons: {', '.join(f'{h}d' for h in horizons)}")
    if pending_counts:
        pending_summary = ", ".join(
            f"{h}d={pending_counts[h]}" for h in sorted(pending_counts) if pending_counts[h] > 0
        )
        if pending_summary:
            print(f"Pending scoring: {pending_summary}")

    improvement = overall["raw_mae"] - overall["model_mae"] if overall else 0
    if improvement > 0:
        print(f"Predictions improve MAE by {improvement:.2f}°F over raw forecast")
    else:
        print(f"Raw forecast beats predictions by {-improvement:.2f}°F")


def run_live_forecast(
    station_id: str,
    data_dir: Path = Path("data"),
    log_dir: Path = Path("runs/live"),
    model_type: str = LIVE_MODEL_TYPE,
    alpha: float = 0.01,
    horizons: list[int] | None = None,
) -> list[dict]:
    """Run a live forecast and return results."""
    if horizons is None:
        horizons = LIVE_HORIZONS
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "predictions.jsonl"
    feature_path = data_dir / "features" / "train_daily_tmax" / f"{station_id}.parquet"

    print(f"[live] Fetching forecast for {station_id}...")
    try:
        forecast_df = fetch_live_forecast(station_id)
    except Exception as e:
        print(f"[live] ERROR: Forecast fetch failed: {e}")
        print_accuracy_report(log_path)
        sys.exit(1)

    # Sanity checks on fetched forecast
    warnings = sanity_check_forecast(forecast_df)
    for w in warnings:
        print(f"[live] WARNING: {w}")
    if any("NaN/Inf" in w for w in warnings):
        print("[live] FATAL: Forecast contains invalid values. Aborting.")
        sys.exit(1)

    print(f"[live] Loading historical data...")
    train_df = load_and_prepare_training_data(data_dir, station_id)

    # Backfill actuals from previous predictions
    train_df, n_backfilled = backfill_actuals(log_path, train_df, station_id, feature_path)

    # Log scored prediction counts (retrain trigger visibility)
    scored_counts = _scored_predictions_by_horizon(log_path)
    if scored_counts:
        counts_str = ", ".join(f"{h}d={scored_counts.get(h, 0)}" for h in horizons)
        print(f"[live] Scored predictions by horizon: {counts_str}")
        for h in horizons:
            if scored_counts.get(h, 0) >= 30:
                print(f"[live] Retrain threshold met: horizon {h}d has {scored_counts[h]} scored predictions")

    today = date.today()

    # Compute horizon_days and filter to desired horizons
    forecast_df["horizon_days"] = forecast_df["target_date_local"].apply(
        lambda td: (pd.Timestamp(td).date() - today).days if hasattr(td, 'date') or isinstance(td, date) else (pd.Timestamp(str(td)).date() - today).days
    )
    forecast_df = forecast_df[forecast_df["horizon_days"].isin(horizons)].reset_index(drop=True)
    print(f"[live] Horizons to predict: {sorted(forecast_df['horizon_days'].tolist())} days")

    print(f"[live] Got {len(forecast_df)} forecast day(s):")
    for _, row in forecast_df.iterrows():
        t = row["target_date_local"]
        t_str = str(t.date() if hasattr(t, "date") else t)
        h = row.get("horizon_days", "?")
        print(f"  {t_str}: raw={row['tmax_pred_f']:.1f}°F, lead={row['lead_hours']}h (h={h}d)")

    # Try loading cached model first
    model = load_model_version(log_dir)
    if model is not None:
        print(f"[live] Loaded cached model version")
        # Still need to verify it's compatible; if train_rows changed significantly, retrain
        model_dir = _get_model_dir(log_dir)
        current_link = model_dir / "current.pkl"
        cached_meta = None
        if current_link.exists() or current_link.is_symlink():
            try:
                meta_path = current_link.resolve().with_suffix(".json")
                if meta_path.exists():
                    with open(meta_path) as f:
                        cached_meta = json.load(f)
            except Exception:
                pass
        if cached_meta and (cached_meta.get("train_rows", 0) < len(train_df) - 5):
            print(f"[live] Cached model stale (trained on {cached_meta.get('train_rows')} rows, current {len(train_df)}). Retraining...")
            model = None

    if model is None:
        print(f"[live] Training on {len(train_df)} rows ({n_backfilled} new)...")
        if len(train_df) < 100:
            print(f"[live] WARNING: Training data only has {len(train_df)} rows (< 100)")
        if model_type == "stacked":
            model = create_forecaster(
                model_type,
                alpha=alpha,
                features=LIVE_FEATURES,
                hyperparams=LIVE_STACKED_HYPERPARAMS,
            )
        else:
            model = create_forecaster(model_type, alpha=alpha, features=LIVE_FEATURES)
        model.fit(train_df)

    # Calibration on recent data
    recent = train_df.tail(90).copy()
    recent["y_pred_f"] = model.predict_mu(recent)
    recent["y_true_f"] = recent["tmax_actual_f"]
    recent_metrics = compute_forecast_metrics(recent)
    recent_bands = compute_accuracy_bands(recent)
    print(f"[live] Recent 90-day calibration: MAE={recent_metrics.mae:.2f}°F, "
          f"±1F={recent_bands.within_1f:.0%}, ±2F={recent_bands.within_2f:.0%}")

    # Predict
    results = []
    for _, fc_row in forecast_df.iterrows():
        live_features = build_live_features(fc_row, train_df)
        lead_hours = int(fc_row["lead_hours"])
        raw = fc_row["tmax_pred_f"]
        horizon_days = int(fc_row["horizon_days"])

        # Select horizon-appropriate features
        if horizon_days <= 2:
            horizon_features = LIVE_FEATURES_BY_HORIZON['short']
        elif horizon_days <= 5:
            horizon_features = LIVE_FEATURES_BY_HORIZON['medium']
        else:
            horizon_features = LIVE_FEATURES_BY_HORIZON['long']
        # Filter live_features to only the relevant features for this horizon
        horizon_feature_cols = [c for c in horizon_features if c in live_features.columns]

        # Use stacked model only for short leads (≤48h); rolling bias correction for longer leads
        should_ensemble = (model_type == "stacked" and lead_hours <= 48)
        if should_ensemble:
            # Pass only horizon-appropriate features to the model
            horizon_df = live_features[horizon_feature_cols].copy()
            # Ensure all features the model expects are present by filling missing
            for col in LIVE_FEATURES:
                if col not in horizon_df.columns:
                    horizon_df[col] = 0.0
            prediction = model.predict_mu(horizon_df)[0]
            model_used = "stacked"
        elif lead_hours > 48:
            # For longer horizons, apply rolling bias correction
            # Compute recent rolling bias from training data
            recent_bias = 0.0
            if len(train_df) >= 14:
                recent = train_df.tail(30)
                if 'tmax_actual_f' in recent.columns and 'tmax_pred_f' in recent.columns:
                    recent_bias = (recent['tmax_pred_f'] - recent['tmax_actual_f']).mean()
            prediction = raw - recent_bias  # subtract bias from raw
            model_used = "bias_corrected"
        else:
            prediction = raw
            model_used = "raw"

        correction = prediction - raw
        target = fc_row["target_date_local"]
        target_str = str(target.date() if hasattr(target, "date") else target)

        # Warning for large corrections
        if abs(correction) > 5.0:
            print(f"[live] WARNING: Large correction for {target_str}: {correction:+.1f}°F")

        result = {
            "date": datetime.now(timezone.utc).isoformat(),
            "target_date": target_str,
            "station_id": station_id,
            "model_type": model_type,
            "feature_set": LIVE_FEATURES,
            "raw_forecast_f": round(raw, 1),
            "model_prediction_f": round(prediction, 1),
            "ridge_prediction_f": round(prediction, 1),
            "correction_f": round(correction, 1),
            "lead_hours": lead_hours,
            "horizon_days": int(fc_row["horizon_days"]),
            "alpha": alpha,
            "train_rows": len(train_df),
            "recent_mae": round(recent_metrics.mae, 2),
            "model_used": model_used,
        }

        # Validate before accepting
        valid, err_msg = validate_prediction(result)
        if not valid:
            print(f"[live] ERROR: Prediction validation failed for {target_str}: {err_msg}")
            continue

        results.append(result)

        print(f"\n[live] === {target_str} (h={int(fc_row['horizon_days'])}d, model={model_used}) ===")
        print(f"  Raw forecast:    {raw:.1f}°F")
        print(f"  Predict ({model_used}): {prediction:.1f}°F  (correction: {correction:+.1f}°F)")

    # Write valid results to log
    if results:
        with open(log_path, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\n[live] Logged {len(results)} prediction(s) to {log_path}")
    else:
        print(f"\n[live] WARNING: No valid predictions generated")
        sys.exit(1)

    # Save model version (only if we trained a new one or significant change)
    save_model_version(
        model=model,
        train_rows=len(train_df),
        station_id=station_id,
        model_type=model_type,
        alpha=alpha,
        horizons=horizons,
        recent_mae=recent_metrics.mae,
        log_dir=log_dir,
    )

    # Print accuracy report if we have scored predictions
    print_accuracy_report(log_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Live daily temperature forecast")
    parser.add_argument("--station", default=None, help="Station ID (e.g., KNYC, KLGA). Defaults to TEMP_DATA_STATION env var or KNYC")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--log-dir", default="runs/live", type=Path)
    parser.add_argument("--model-type", choices=["ridge", "stacked"], default=LIVE_MODEL_TYPE)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--horizons", type=int, nargs="+", default=LIVE_HORIZONS,
                        help="Forecast horizons in days (default: 1 2 3 5 7)")
    parser.add_argument("--report", action="store_true", help="Show accuracy history and exit")
    args = parser.parse_args()

    horizons = sorted(args.horizons)
    
    # Resolve station: CLI arg > env var > default
    station_id = args.station or os.getenv("TEMP_DATA_STATION", "KNYC")

    log_path = args.log_dir / "predictions.jsonl"
    if args.report:
        print_accuracy_report(log_path)
        return

    run_live_forecast(
        station_id=station_id,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        model_type=args.model_type,
        alpha=args.alpha,
        horizons=horizons,
    )


if __name__ == "__main__":
    main()
