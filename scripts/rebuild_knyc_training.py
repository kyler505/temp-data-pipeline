#!/usr/bin/env python3
"""Rebuild KNYC feature data from predictions log (fast recovery)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

log_path = Path("runs/live/predictions.jsonl")
lines = log_path.read_text().strip().split("\n")
preds = [json.loads(l) for l in lines if l.strip()]

# Deduplicate: keep latest per (target_date, horizon_days)
by_key = {}
for p in preds:
    k = (p["target_date"], p.get("horizon_days", 1))
    existing = by_key.get(k)
    if existing is None or p.get("date", "") > existing.get("date", ""):
        by_key[k] = p

rows = []
for p in by_key.values():
    td = pd.Timestamp(p["target_date"])
    rows.append({
        "station_id": "KNYC",
        "target_date_local": td,
        "tmax_pred_f": p["raw_forecast_f"],
        "tmax_actual_f": p.get("actual_f"),
        "lead_hours": p.get("lead_hours", 24),
        "month": td.month,
        "sin_doy": np.sin(2 * np.pi * td.dayofyear / 366),
        "cos_doy": np.cos(2 * np.pi * td.dayofyear / 366),
    })

df = pd.DataFrame(rows).sort_values("target_date_local").reset_index(drop=True)

# Compute rolling bias features
df["bias"] = df["tmax_pred_f"] - df["tmax_actual_f"]
df["bias_7d"] = df["bias"].rolling(7, min_periods=3).mean().fillna(0)
df["bias_14d"] = df["bias"].rolling(14, min_periods=5).mean().fillna(0)
df["bias_30d"] = df["bias"].rolling(30, min_periods=7).mean().fillna(0)
df["rmse_14d"] = (df["bias"] ** 2).rolling(14, min_periods=5).mean().apply(np.sqrt).fillna(1.0)
df["rmse_30d"] = (df["bias"] ** 2).rolling(30, min_periods=7).mean().apply(np.sqrt).fillna(1.0)
df["sigma_lead"] = df.groupby("lead_hours")["bias"].transform(lambda x: x.expanding().std()).fillna(1.0)

# Drop null actuals and bias column
df = df.dropna(subset=["tmax_actual_f"]).drop(columns=["bias"])

out_dir = Path("data/features/train_daily_tmax")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "KNYC.parquet"
df.to_parquet(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
print(f"Date range: {df.target_date_local.min()} to {df.target_date_local.max()}")
print(f"Columns: {list(df.columns)}")
