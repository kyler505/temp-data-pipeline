#!/usr/bin/env python3
"""Expanded XGBoost tuning sweep on enriched features (KLGA).

Focuses the search around the best known config (lr=0.05, depth=3, n=300, reg=0.1)
with wider ranges and finer granularity.
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
OUTPUT_DIR = Path("runs/enriched_tuning_v2")

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
ENRICHED_ALL = BASELINE_FEATURES + ENRICHED_FEATURES


def expanded_xgboost_grid():
    """Wider grid focused around the best known params."""
    # Best known: lr=0.05, depth=3, n=300, reg=0.1
    for lr, depth, n_est, reg in product(
        [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],  # finer lr
        [2, 3, 4, 5, 6],                               # more depths
        [200, 300, 500, 750, 1000],                     # higher n_estimators
        [0.0, 0.05, 0.1, 0.5, 1.0],                    # finer regularization
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
    """Walk-forward eval on last 3 folds for speed."""
    config = EvalConfig(
        run_name="enriched_tuning_v2",
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

    enriched_mask = (df["pressure_mean"] > 0) & (df["humidity_mean"] > 0)
    df = df[enriched_mask].reset_index(drop=True)

    wf = WalkForwardSplit(window_size=730, step_size=90, expanding=True)
    folds = wf.generate_folds(df)
    if not folds:
        return None

    fold_metrics = []
    for train_df, test_df in folds[-3:]:
        model = create_forecaster(
            model_cfg["type"],
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
                "mae": fm.mae, "rmse": fm.rmse, "bias": fm.bias,
                "r2": fm.r2, "within_1f": ab.within_1f, "within_2f": ab.within_2f,
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
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    enriched_path = Path(DATA_DIR) / "train" / "daily_tmax" / STATION / "train_daily_tmax_enriched.parquet"
    print(f"[tune] Loading enriched features from {enriched_path}")
    feature_df = pd.read_parquet(enriched_path)
    print(f"[tune] Loaded {len(feature_df)} rows")

    best_mae = float("inf")
    best_result = None
    count = 0
    all_results = []

    grid = list(expanded_xgboost_grid())
    print(f"[tune] Expanded grid: {len(grid)} configs × 3 folds = {len(grid) * 3} evaluations")
    print(f"[tune] === Sweeping expanded XGBoost on {len(ENRICHED_ALL)} enriched features ===")

    for model_cfg in grid:
        count += 1
        hp = model_cfg["hyperparams"]
        params_str = f"lr={hp['learning_rate']}, depth={hp['max_depth']}, n={hp['n_estimators']}, reg={hp['reg_alpha']}"

        result = evaluate_config(feature_df, model_cfg, ENRICHED_ALL)
        if result is None:
            print(f"  [{count}/{len(grid)}] {params_str} -> FAILED")
            continue

        mae = result["mae"]
        flag = ""
        if mae < best_mae:
            best_mae = mae
            best_result = {**result, "params": model_cfg}
            flag = " ★ NEW BEST"

        if flag or count % 50 == 0:
            print(f"  [{count}/{len(grid)}] {params_str} -> MAE={mae:.3f} RMSE={result['rmse']:.3f} ±1F={result['within_1f']:.1%}{flag}")

        all_results.append({"config_num": count, "params": model_cfg, **result})

    # Write all results
    output_path = OUTPUT_DIR / "enriched_tuning_v2_results.json"
    output_data = {
        "best": best_result,
        "n_configs": len(all_results),
        "results": sorted(all_results, key=lambda x: x["mae"])[:20],  # top 20
    }
    output_path.write_text(json.dumps(output_data, indent=2, default=str))

    # Also write just the top 20 for easy reading
    top_path = OUTPUT_DIR / "top20.json"
    top_path.write_text(json.dumps(sorted(all_results, key=lambda x: x["mae"])[:20], indent=2, default=str))

    print(f"\n[tune] Results written to {output_path}")
    print(f"[tune] Top 20 written to {top_path}")

    if best_result:
        hp = best_result["params"]["hyperparams"]
        print(f"\n[tune] BEST: MAE={best_result['mae']:.3f} RMSE={best_result['rmse']:.3f} "
              f"±1F={best_result['within_1f']:.1%}")
        print(f"  lr={hp['learning_rate']}, depth={hp['max_depth']}, "
              f"n={hp['n_estimators']}, reg={hp['reg_alpha']}")

    # Top 10 summary
    print("\n" + "=" * 90)
    print(f"TOP 10 ENRICHED XGBOOST CONFIGS (out of {count} tested)")
    print("=" * 90)
    print(f"{'#':>3} {'MAE':>6} {'RMSE':>6} {'±1F%':>6} {'±2F%':>6} {'lr':>5} {'depth':>5} {'n_est':>5} {'reg':>5}")
    print("-" * 90)
    top10 = sorted(all_results, key=lambda x: x["mae"])[:10]
    for i, r in enumerate(top10):
        hp = r["params"]["hyperparams"]
        print(f"{i+1:>3} {r['mae']:>6.3f} {r['rmse']:>6.3f} {r['within_1f']:>5.1%} {r['within_2f']:>5.1%} "
              f"{hp['learning_rate']:>5} {hp['max_depth']:>5} {hp['n_estimators']:>5} {hp['reg_alpha']:>5}")


if __name__ == "__main__":
    main()
