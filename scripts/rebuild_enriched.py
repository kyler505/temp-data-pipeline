#!/usr/bin/env python3
"""Rebuild the training feature table with enriched Open-Meteo variables.

Usage:
    python scripts/rebuild_enriched.py [--station KLGA] [--start 2009-01-01] [--end 2024-12-31]
"""
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, "src")

from tempdata.features.enriched_features import (
    fetch_enriched_in_chunks,
    merge_enriched_features,
)
from tempdata.fetch.noaa_hourly import resolve_station


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default="KLGA")
    parser.add_argument("--start", default="2009-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--data-dir", default="data", type=Path)
    args = parser.parse_args()

    station = resolve_station(args.station)
    feature_path = args.data_dir / "train" / "daily_tmax" / args.station / "train_daily_tmax.parquet"
    cache_dir = args.data_dir / "cache" / "openmeteo_enriched"

    print(f"[rebuild] Loading existing features from {feature_path}")
    train_df = pd.read_parquet(feature_path) if feature_path.exists() else None
    if train_df is None:
        print("[rebuild] ERROR: no existing training data found")
        sys.exit(1)

    print(f"[rebuild] {len(train_df)} rows in training data")
    print(f"[rebuild] Fetching enriched Open-Meteo data ({args.start} to {args.end})...")

    enriched_df = fetch_enriched_in_chunks(
        lat=station.lat,
        lon=station.lon,
        start_date=args.start,
        end_date=args.end,
        tz=station.tz,
        chunk_days=365,
        cache_dir=cache_dir,
    )

    print(f"[rebuild] Fetched {len(enriched_df)} enriched rows")

    print("[rebuild] Merging features...")
    merged = merge_enriched_features(train_df, enriched_df)

    print(f"[rebuild] Result: {len(merged)} rows, {len(merged.columns)} columns")
    new_cols = [c for c in merged.columns if c not in train_df.columns]
    print(f"[rebuild] New columns: {new_cols}")

    # Save enriched training table
    enriched_path = args.data_dir / "train" / "daily_tmax" / args.station / "train_daily_tmax_enriched.parquet"
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(enriched_path, index=False)
    print(f"[rebuild] Saved to {enriched_path}")


import pandas as pd
if __name__ == "__main__":
    main()
