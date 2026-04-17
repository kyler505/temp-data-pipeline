"""Feature ablation study for stacked ensemble on KLGA temperature forecast.

Usage:
    python -m tempdata.ablate                          # defaults
    python -m tempdata.ablate --data-path <path>       # custom data
    python -m tempdata.ablate --model stacked          # which model to ablate

Runs leave-one-out and leave-one-group-out ablation on a time-based
validation split.  Reports RMSE/MAE change vs. full-feature baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

CORE_FEATURES = ["tmax_pred_f", "sin_doy", "cos_doy", "lead_hours"]

ROLLING_FEATURES = ["bias_7d", "bias_14d", "bias_30d", "rmse_14d", "rmse_30d", "sigma_lead"]

ENRICHED_FEATURES = [
    "tmin_f", "apparent_tmax_f", "diurnal_range_f",
    "wind_max_kmh", "wind_u", "wind_v",
    "precip_mm", "precip_hours",
    "radiation_mj",
    "humidity_mean", "humidity_range",
    "pressure_mean", "pressure_trend",
    "cloud_cover_mean", "dew_point_mean_f",
]

ALL_GROUPS = {
    "core": CORE_FEATURES,
    "rolling": ROLLING_FEATURES,
    "enriched": ENRICHED_FEATURES,
}


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_training_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"])
    df = df.sort_values("target_date_local").reset_index(drop=True)
    return df


def temporal_split(
    df: pd.DataFrame, train_frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple train/val split by time fraction."""
    n = len(df)
    split_idx = int(n * train_frac)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(model_type: str, features: list[str], hyperparams: dict | None = None):
    """Build a forecaster for ablation."""
    from tempdata.eval.models import create_forecaster
    return create_forecaster(
        model_type=model_type,
        features=features,
        hyperparams=hyperparams or {},
    )


def train_and_score(
    model_type: str,
    features: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: dict | None = None,
) -> dict[str, float]:
    """Train model on features and return val-set metrics."""
    # Filter to features that exist
    avail = [f for f in features if f in train_df.columns]
    if not avail:
        return {"mae": np.nan, "rmse": np.nan, "n_features": 0, "error": "no features"}

    t0 = time.time()
    try:
        model = build_model(model_type, avail, hyperparams)
        model.fit(train_df)
        preds = model.predict_mu(val_df)
        actuals = val_df["tmax_actual_f"].values
        errors = np.abs(preds - actuals)
        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
    except Exception as e:
        return {"mae": np.nan, "rmse": np.nan, "n_features": len(avail), "error": str(e)}

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "n_features": len(avail),
        "train_time_s": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Ablation strategies
# ---------------------------------------------------------------------------

def run_leave_one_out(
    model_type: str,
    all_features: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: dict | None = None,
) -> list[dict]:
    """Remove one feature at a time, measure impact."""
    # Baseline: all features
    baseline = train_and_score(model_type, all_features, train_df, val_df, hyperparams)
    results = []

    for feat in all_features:
        reduced = [f for f in all_features if f != feat]
        score = train_and_score(model_type, reduced, train_df, val_df, hyperparams)
        results.append({
            "removed_feature": feat,
            "rmse": score["rmse"],
            "mae": score["mae"],
            "rmse_delta": round(score["rmse"] - baseline["rmse"], 4) if np.isfinite(score["rmse"]) else np.nan,
            "mae_delta": round(score["mae"] - baseline["mae"], 4) if np.isfinite(score["mae"]) else np.nan,
            "n_features": score["n_features"],
        })

    # Sort by RMSE delta (negative = removing feature improved model)
    results.sort(key=lambda r: r["rmse_delta"] if np.isfinite(r["rmse_delta"]) else 1e9)
    return results


def run_leave_one_group_out(
    model_type: str,
    all_features: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: dict | None = None,
) -> list[dict]:
    """Remove one feature group at a time, measure impact."""
    baseline = train_and_score(model_type, all_features, train_df, val_df, hyperparams)
    results = []

    for group_name, group_feats in ALL_GROUPS.items():
        reduced = [f for f in all_features if f not in group_feats]
        if not reduced:
            continue
        score = train_and_score(model_type, reduced, train_df, val_df, hyperparams)
        results.append({
            "removed_group": group_name,
            "removed_features": [f for f in group_feats if f in all_features],
            "rmse": score["rmse"],
            "mae": score["mae"],
            "rmse_delta": round(score["rmse"] - baseline["rmse"], 4) if np.isfinite(score["rmse"]) else np.nan,
            "mae_delta": round(score["mae"] - baseline["mae"], 4) if np.isfinite(score["mae"]) else np.nan,
            "n_features": score["n_features"],
        })

    results.sort(key=lambda r: r["rmse_delta"] if np.isfinite(r["rmse_delta"]) else 1e9)
    return results


def run_greedy_forward(
    model_type: str,
    candidate_features: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: dict | None = None,
) -> list[dict]:
    """Greedy forward selection: start from empty, add best feature each round."""
    selected: list[str] = []
    remaining = list(candidate_features)
    history = []

    # Score empty model first (will fail, just record it)
    baseline_score = train_and_score(model_type, [], train_df, val_df, hyperparams)
    history.append({
        "step": 0,
        "added_feature": None,
        "selected": [],
        "rmse": baseline_score["rmse"],
        "mae": baseline_score["mae"],
    })

    step = 0
    while remaining:
        step += 1
        best_feat = None
        best_rmse = np.inf

        for feat in remaining:
            trial_feats = selected + [feat]
            score = train_and_score(model_type, trial_feats, train_df, val_df, hyperparams)
            if np.isfinite(score["rmse"]) and score["rmse"] < best_rmse:
                best_rmse = score["rmse"]
                best_feat = feat

        if best_feat is None:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        score = train_and_score(model_type, selected, train_df, val_df, hyperparams)
        history.append({
            "step": step,
            "added_feature": best_feat,
            "selected": list(selected),
            "rmse": score["rmse"],
            "mae": score["mae"],
            "n_features": len(selected),
        })

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Feature ablation study")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/train/daily_tmax/KLGA/train_daily_tmax_enriched.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ablate"))
    parser.add_argument("--model", type=str, default="ridge", help="Model type to ablate")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--strategy", choices=["loo", "loog", "forward", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="Skip forward selection (slow)")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}")
    df = load_training_data(args.data_path)
    train_df, val_df = temporal_split(df, train_frac=args.train_frac)
    print(f"Data: {len(df)} rows")
    print(f"Train: {len(train_df)} ({train_df['target_date_local'].min().date()}..{train_df['target_date_local'].max().date()})")
    print(f"Val:   {len(val_df)} ({val_df['target_date_local'].min().date()}..{val_df['target_date_local'].max().date()})")

    # All available features
    all_features = CORE_FEATURES + ROLLING_FEATURES + [f for f in ENRICHED_FEATURES if f in df.columns]
    all_features = [f for f in all_features if f in df.columns]
    print(f"Features ({len(all_features)}): {all_features}")
    print(f"Model: {args.model}")
    print()

    # Baseline
    baseline = train_and_score(args.model, all_features, train_df, val_df)
    print(f"Baseline (all features): MAE={baseline['mae']:.4f}  RMSE={baseline['rmse']:.4f}")
    print()

    output = {
        "run_id": date.today().isoformat() + "_ablate",
        "model": args.model,
        "data_path": str(args.data_path),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "all_features": all_features,
        "baseline": baseline,
    }

    strategy = args.strategy
    if args.quick and strategy == "all":
        strategy = "loo"  # skip forward

    # Leave-one-out
    if strategy in ("loo", "all"):
        print("=== Leave-One-Out Ablation ===")
        loo_results = run_leave_one_out(args.model, all_features, train_df, val_df)
        output["leave_one_out"] = loo_results

        print(f"\n{'Feature':<25} {'RMSE':>8} {'Δ RMSE':>8} {'MAE':>8} {'Δ MAE':>8}")
        print("-" * 60)
        for r in loo_results:
            delta_rmse = f"{r['rmse_delta']:+.4f}" if np.isfinite(r['rmse_delta']) else "N/A"
            delta_mae = f"{r['mae_delta']:+.4f}" if np.isfinite(r['mae_delta']) else "N/A"
            print(f"{r['removed_feature']:<25} {r['rmse']:>8.4f} {delta_rmse:>8} {r['mae']:>8.4f} {delta_mae:>8}")
        print()

        # Highlight features that hurt when removed
        negative_impact = [r for r in loo_results if np.isfinite(r['rmse_delta']) and r['rmse_delta'] > 0.1]
        if negative_impact:
            print("Features that MATTER (removing increases RMSE by >0.1):")
            for r in negative_impact:
                print(f"  {r['removed_feature']}: +{r['rmse_delta']:.4f} RMSE")
            print()

        # Highlight features that don't help
        no_impact = [r for r in loo_results if np.isfinite(r['rmse_delta']) and r['rmse_delta'] <= 0.01]
        if no_impact:
            print("Features that DON'T MATTER (removing changes RMSE by ≤0.01):")
            for r in no_impact:
                print(f"  {r['removed_feature']}: {r['rmse_delta']:+.4f} RMSE")
            print()

    # Leave-one-group-out
    if strategy in ("loog", "all"):
        print("=== Leave-One-Group-Out Ablation ===")
        loog_results = run_leave_one_group_out(args.model, all_features, train_df, val_df)
        output["leave_one_group_out"] = loog_results

        print(f"\n{'Group':<15} {'RMSE':>8} {'Δ RMSE':>8} {'MAE':>8} {'Δ MAE':>8}  Features removed")
        print("-" * 80)
        for r in loog_results:
            delta_rmse = f"{r['rmse_delta']:+.4f}" if np.isfinite(r['rmse_delta']) else "N/A"
            delta_mae = f"{r['mae_delta']:+.4f}" if np.isfinite(r['mae_delta']) else "N/A"
            removed = ", ".join(r["removed_features"][:4])
            if len(r["removed_features"]) > 4:
                removed += f" +{len(r['removed_features'])-4} more"
            print(f"{r['removed_group']:<15} {r['rmse']:>8.4f} {delta_rmse:>8} {r['mae']:>8.4f} {delta_mae:>8}  {removed}")
        print()

    # Greedy forward selection
    if strategy in ("forward", "all") and not args.quick:
        print("=== Greedy Forward Selection ===")
        forward_results = run_greedy_forward(args.model, all_features, train_df, val_df)
        output["forward_selection"] = forward_results

        print(f"\n{'Step':>4} {'Added Feature':<25} {'RMSE':>8} {'MAE':>8} {'# Feat':>6}")
        print("-" * 55)
        for r in forward_results:
            feat = r["added_feature"] or "(none)"
            print(f"{r['step']:>4} {feat:<25} {r['rmse'] or 0:>8.4f} {r['mae'] or 0:>8.4f} {r.get('n_features', 0):>6}")
        print()

    # Write output
    out_dir = args.output_dir / output["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"Results written to {summary_path}")

    # Recommendation
    if "leave_one_out" in output:
        important = [r for r in output["leave_one_out"] if np.isfinite(r["rmse_delta"]) and r["rmse_delta"] > 0.05]
        unimportant = [r for r in output["leave_one_out"] if np.isfinite(r["rmse_delta"]) and r["rmse_delta"] <= 0.0]
        print("\n--- RECOMMENDATION ---")
        if important:
            print(f"KEEP ({len(important)} features with >0.05 RMSE impact):")
            for r in sorted(important, key=lambda x: -x["rmse_delta"]):
                print(f"  {r['removed_feature']}: +{r['rmse_delta']:.4f}")
        if unimportant:
            print(f"CONSIDER REMOVING ({len(unimportant)} features with ≤0 impact):")
            for r in unimportant:
                print(f"  {r['removed_feature']}: {r['rmse_delta']:+.4f}")


if __name__ == "__main__":
    main()
