#!/usr/bin/env python3
"""Hyperparameter tuning sweep for XGBoost on enriched features (KLGA).

Extends the baseline tune_sweep to evaluate XGBoost with enriched meteorological
features (humidity, wind, pressure, cloud cover, etc.).

Also runs Ridge (alpha=100) on enriched features as a comparison baseline.
"""
import json
import sys
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from tempdata.eval.config import EvalConfig, SplitConfig
from tempdata.eval.data import _normalize_eval_dates, _apply_filters, _ensure_features
from tempdata.eval.models import create_forecaster
from tempdata.eval.splits import WalkForwardSplit
from tempdata.eval.metrics import compute_forecast_metrics, compute_accuracy_bands


DATA_DIR = "data"
STATION = "KLGA"
START = date(2017, 1, 1)
END = date(2024, 12, 31)
OUTPUT_DIR = Path("runs/enriched_tuning")

ENRICHED_FEATURES = [
    "tmin_f", "apparent_tmax_f", "diurnal_range_f",
    "wind_max_kmh", "wind_u", "wind_v",
    "precip_mm", "precip_hours",
    "radiation_mj",
    "humidity_mean", "humidity_range",
    "pressure_mean", "pressure_trend",
    "cloud_cover_mean", "dew_point_mean_f",
]

BASELINE_FEATURES = ["tmax_pred_f", "sin_doy", "cos_doy", "bias_7d", "bias_14d", "lead_hours"]
ENRICHED_ALL = BASELINE_FEATURES + ENRICHED_FEATURES  # 21 features


def xgboost_grid():
    for lr, depth, n_est, reg in product(
        [0.01, 0.05, 0.1],
        [2, 3, 5, 7],
        [100, 300, 500],
        [0.0, 0.1, 1.0],
    ):
        yield {
            "type": "xgboost",
            "hyperparams": {
                "learning_rate": lr,
                "max_depth": depth,
                "n_estimators": n_est,
                "reg_alpha": reg,
            },
        }


def evaluate_config(feature_df, model_cfg, features):
    """Walk-forward eval on last 3 folds for speed. Returns avg metrics or None."""
    from tempdata.eval.data import _normalize_eval_dates, _apply_filters, _ensure_features

    config = EvalConfig(
        run_name="enriched_tuning",
        station_ids=[STATION],
        start_date_local=START,
        end_date_local=END,
        split=SplitConfig(type="walk_forward", window_size=730, step_size=90),
    )

    df = feature_df.copy()
    df = _normalize_eval_dates(df)
    df = _apply_filters(df, config)
    df = _ensure_features(df)
    df = df.sort_values(["station_id", "target_date_local"]).reset_index(drop=True)

    # Filter to rows where enriched features are available
    enriched_mask = (df["pressure_mean"] > 0) & (df["humidity_mean"] > 0)
    df = df[enriched_mask].reset_index(drop=True)

    wf = WalkForwardSplit(window_size=730, step_size=90, expanding=True)
    folds = wf.generate_folds(df)
    if not folds:
        return None

    # Use last 3 folds for speed
    fold_metrics = []
    for train_df, test_df in folds[-3:]:
        model = create_forecaster(
            model_cfg["type"],
            alpha=model_cfg.get("alpha", 1.0),
            features=features,
            hyperparams=model_cfg.get("hyperparams"),
        )
        try:
            model.fit(train_df)
            preds = model.predict_mu(test_df)
            test_eval = test_df.copy()
            test_eval["y_pred_f"] = preds
            test_eval["y_true_f"] = test_eval["tmax_actual_f"]

            fm = compute_forecast_metrics(test_eval)
            ab = compute_accuracy_bands(test_eval)
            fold_metrics.append({
                "mae": fm.mae,
                "rmse": fm.rmse,
                "bias": fm.bias,
                "r2": fm.r2,
                "within_1f": ab.within_1f,
                "within_2f": ab.within_2f,
            })
        except Exception as e:
            fold_metrics.append({"error": str(e)})

    valid = [m for m in fold_metrics if "error" not in m]
    if not valid:
        return None

    return {
        "mae": np.mean([m["mae"] for m in valid]),
        "rmse": np.mean([m["rmse"] for m in valid]),
        "bias": np.mean([m["bias"] for m in valid]),
        "r2": np.mean([m["r2"] for m in valid]),
        "within_1f": np.mean([m["within_1f"] for m in valid]),
        "within_2f": np.mean([m["within_2f"] for m in valid]),
        "n_folds": len(valid),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load enriched data
    enriched_path = Path(DATA_DIR) / "train" / "daily_tmax" / STATION / "train_daily_tmax_enriched.parquet"
    print(f"[tune] Loading enriched features from {enriched_path}")
    feature_df = pd.read_parquet(enriched_path)
    print(f"[tune] Loaded {len(feature_df)} rows")

    # Quick data check
    enriched_mask = (feature_df["pressure_mean"] > 0) & (feature_df["humidity_mean"] > 0)
    print(f"[tune] Rows with enriched features available: {enriched_mask.sum()}")

    all_results = []

    # ── Ridge baseline on enriched (reference point) ──────────────────
    print("\n[tune] === Ridge (alpha=100) on enriched features (baseline) ===")
    ridge_cfg = {"type": "ridge", "alpha": 100.0}
    result = evaluate_config(feature_df, ridge_cfg, ENRICHED_ALL)
    if result:
        entry = {
            "model": "ridge_enriched",
            "best_mae": result["mae"],
            "best_rmse": result["rmse"],
            "best_bias": result["bias"],
            "best_r2": result["r2"],
            "within_1f": result["within_1f"],
            "within_2f": result["within_2f"],
            "params": ridge_cfg,
        }
        all_results.append(entry)
        print(f"  MAE={result['mae']:.3f} RMSE={result['rmse']:.3f} ±1F={result['within_1f']:.1%}")

    # ── XGBoost sweep on enriched features ────────────────────────────
    print(f"\n[tune] === Sweeping XGBoost on {len(ENRICHED_ALL)} enriched features ===")
    best_mae = float("inf")
    best_result = None
    count = 0

    for model_cfg in xgboost_grid():
        count += 1
        hp = model_cfg["hyperparams"]
        params_str = f"lr={hp['learning_rate']}, depth={hp['max_depth']}, n={hp['n_estimators']}, reg={hp['reg_alpha']}"

        result = evaluate_config(feature_df, model_cfg, ENRICHED_ALL)
        if result is None:
            print(f"  [{count}] {params_str} -> FAILED")
            continue

        mae = result["mae"]
        flag = ""
        if mae < best_mae:
            best_mae = mae
            best_result = {**result, "params": model_cfg}
            flag = " ★ NEW BEST"

        print(f"  [{count}] {params_str} -> MAE={mae:.3f} RMSE={result['rmse']:.3f} ±1F={result['within_1f']:.1%}{flag}")

    if best_result:
        entry = {
            "model": "xgboost_enriched",
            "best_mae": best_result["mae"],
            "best_rmse": best_result["rmse"],
            "best_bias": best_result["bias"],
            "best_r2": best_result["r2"],
            "within_1f": best_result["within_1f"],
            "within_2f": best_result["within_2f"],
            "params": best_result["params"],
            "n_configs_tested": count,
        }
        all_results.append(entry)
        print(f"[tune] Best enriched XGBoost: MAE={best_result['mae']:.3f}, params={best_result['params']}")

    # ── Write results ─────────────────────────────────────────────────
    output_path = OUTPUT_DIR / "enriched_tuning_results.json"
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n[tune] Results written to {output_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("ENRICHED FEATURES TUNING RESULTS (walk-forward, last 3 folds)")
    print("=" * 80)
    print(f"{'Model':<22} {'MAE':>6} {'RMSE':>6} {'±1F%':>6} {'±2F%':>6} {'Best Params'}")
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: x["best_mae"]):
        p = r["params"]
        pstr = json.dumps(p.get("hyperparams", {"alpha": p.get("alpha")}))
        print(f"{r['model']:<22} {r['best_mae']:>6.3f} {r['best_rmse']:>6.3f} {r['within_1f']:>5.1%} {r['within_2f']:>5.1%} {pstr}")


if __name__ == "__main__":
    main()
