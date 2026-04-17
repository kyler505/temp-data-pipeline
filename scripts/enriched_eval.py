#!/usr/bin/env python3
"""Walk-forward evaluation with enriched features on KLGA.

Compares baseline (original features) vs enriched (15 new meteorological features)
for all models.
"""
import json
import sys
from datetime import date
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
OUTPUT_DIR = Path("runs/enriched_eval")

ENRICHED_FEATURES = [
    "tmin_f", "apparent_tmax_f", "diurnal_range_f",
    "wind_max_kmh", "wind_u", "wind_v",
    "precip_mm", "precip_hours",
    "radiation_mj",
    "humidity_mean", "humidity_range",
    "pressure_mean", "pressure_trend",
    "cloud_cover_mean", "dew_point_mean_f",
]

MODEL_CONFIGS = [
    # Baseline models (original features)
    {"name": "ridge_baseline", "type": "ridge", "alpha": 100.0, "features": None},
    {"name": "xgb_baseline", "type": "xgboost", "hyperparams": {"learning_rate": 0.05, "max_depth": 3, "n_estimators": 100}},
    # Enriched models (all features)
    {"name": "ridge_enriched", "type": "ridge", "alpha": 100.0, "use_enriched": True},
    {"name": "xgb_enriched", "type": "xgboost", "hyperparams": {"learning_rate": 0.05, "max_depth": 5, "n_estimators": 300}, "use_enriched": True},
    {"name": "knn_enriched", "type": "knn", "hyperparams": {"n_neighbors": 15}, "use_enriched": True},
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load enriched data
    enriched_path = Path(DATA_DIR) / "train" / "daily_tmax" / STATION / "train_daily_tmax_enriched.parquet"
    print(f"[eval] Loading enriched features from {enriched_path}")
    feature_df = pd.read_parquet(enriched_path)

    config = EvalConfig(
        run_name="enriched_eval",
        station_ids=[STATION],
        start_date_local=START,
        end_date_local=END,
        split=SplitConfig(type="walk_forward", window_size=730, step_size=90),
    )

    # Prepare data
    df = feature_df.copy()
    df = _normalize_eval_dates(df)
    df = _apply_filters(df, config)
    df = _ensure_features(df)
    df = df.sort_values(["station_id", "target_date_local"]).reset_index(drop=True)

    # Only keep rows where enriched features are available (non-zero)
    enriched_mask = (df["pressure_mean"] > 0) & (df["humidity_mean"] > 0)
    df = df[enriched_mask].reset_index(drop=True)
    print(f"[eval] Dataset: {len(df)} rows (enriched features available)")

    wf = WalkForwardSplit(window_size=730, step_size=90, expanding=True)
    folds = wf.generate_folds(df)
    print(f"[eval] {len(folds)} folds")

    all_results = []

    for mc in MODEL_CONFIGS:
        model_name = mc["name"]
        use_enriched = mc.get("use_enriched", False)

        # Build feature list
        if use_enriched:
            features = ["tmax_pred_f", "sin_doy", "cos_doy", "bias_7d", "bias_14d", "lead_hours"] + ENRICHED_FEATURES
        else:
            features = mc.get("features")

        print(f"\n[eval] === {model_name} ({'enriched' if use_enriched else 'baseline'}) ===")

        fold_results = []
        for i, (train_df, test_df) in enumerate(folds):
            model = create_forecaster(
                mc["type"],
                alpha=mc.get("alpha", 1.0),
                features=features,
                hyperparams=mc.get("hyperparams"),
            )
            try:
                model.fit(train_df)
                preds = model.predict_mu(test_df)
                test_eval = test_df.copy()
                test_eval["y_pred_f"] = preds
                test_eval["y_true_f"] = test_eval["tmax_actual_f"]

                fm = compute_forecast_metrics(test_eval)
                ab = compute_accuracy_bands(test_eval)

                fold_results.append({
                    "fold": i + 1,
                    "mae": fm.mae,
                    "rmse": fm.rmse,
                    "bias": fm.bias,
                    "r2": fm.r2,
                    "within_1f": ab.within_1f,
                    "within_2f": ab.within_2f,
                })
            except Exception as e:
                fold_results.append({"fold": i + 1, "error": str(e)})

        valid = [f for f in fold_results if "error" not in f]
        if valid:
            summary = {
                "model": model_name,
                "enriched": use_enriched,
                "n_features": len(features) if features else "default",
                "n_folds": len(valid),
                "avg_mae": np.mean([f["mae"] for f in valid]),
                "avg_rmse": np.mean([f["rmse"] for f in valid]),
                "avg_bias": np.mean([f["bias"] for f in valid]),
                "avg_r2": np.mean([f["r2"] for f in valid]),
                "avg_within_1f": np.mean([f["within_1f"] for f in valid]),
                "avg_within_2f": np.mean([f["within_2f"] for f in valid]),
                "std_mae": np.std([f["mae"] for f in valid]),
                "folds": fold_results,
            }
            print(f"  MAE={summary['avg_mae']:.3f}±{summary['std_mae']:.3f} "
                  f"RMSE={summary['avg_rmse']:.3f} ±1F={summary['avg_within_1f']:.1%} "
                  f"({summary['n_features']} features)")
        else:
            summary = {"model": model_name, "n_folds": 0, "error": "all failed"}

        all_results.append(summary)

    # Write results
    output_path = OUTPUT_DIR / "enriched_results.json"
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n[eval] Results → {output_path}")

    # Summary
    print("\n" + "=" * 85)
    print("ENRICHED FEATURES EVALUATION (walk-forward, expanding, 90-day steps)")
    print("=" * 85)
    print(f"{'Model':<18} {'Feats':>5} {'MAE':>6} {'±std':>6} {'RMSE':>6} {'±1F%':>6} {'±2F%':>6}")
    print("-" * 85)
    for r in sorted(all_results, key=lambda x: x.get("avg_mae", 999)):
        if "error" in r and r.get("n_folds", 0) == 0:
            print(f"{r['model']:<18} FAILED")
            continue
        nf = r.get("n_features", "?")
        print(f"{r['model']:<18} {nf:>5} {r['avg_mae']:>6.3f} {r['std_mae']:>6.3f} "
              f"{r['avg_rmse']:>6.3f} {r['avg_within_1f']:>5.1%} {r['avg_within_2f']:>5.1%}")


if __name__ == "__main__":
    main()
