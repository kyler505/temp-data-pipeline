"""CLI wrapper for daily Tmax aggregation.

Usage:
    python scripts/build_daily_tmax.py --station KLGA --timezone America/New_York

This reads cleaned hourly data from:
    data/clean/hourly_obs/<station>/*.parquet

And writes daily Tmax to:
    data/clean/daily_tmax/<station>.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tempdata.aggregate.build_daily_tmax import build_daily_tmax, write_daily_tmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build daily Tmax from cleaned hourly observations."
    )
    parser.add_argument(
        "--station",
        required=True,
        help="Station ID (e.g., KLGA)",
    )
    parser.add_argument(
        "--timezone",
        required=True,
        help="Station timezone (e.g., America/New_York)",
    )
    parser.add_argument(
        "--min-coverage",
        type=int,
        default=18,
        help="Minimum hourly observations for a 'good' day (default: 18)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/clean/hourly_obs"),
        help="Base directory for hourly input (default: data/clean/hourly_obs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clean/daily_tmax"),
        help="Directory for daily output (default: data/clean/daily_tmax)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Construct paths
    input_dir = args.input_dir / args.station
    output_path = args.output_dir / f"{args.station}.parquet"

    # Find all hourly parquet files
    hourly_files = sorted(input_dir.glob("*.parquet"))

    if not hourly_files:
        print(f"[aggregate] ERROR: No parquet files found in {input_dir}")
        sys.exit(1)

    print(f"[aggregate] Found {len(hourly_files)} hourly files for {args.station}")

    # Concatenate all hourly partitions
    dfs = []
    for f in hourly_files:
        dfs.append(pd.read_parquet(f))
    hourly_df = pd.concat(dfs, ignore_index=True)

    print(f"[aggregate] Loaded {len(hourly_df)} hourly observations")

    # Build daily Tmax
    daily_df = build_daily_tmax(
        hourly_df,
        station_tz=args.timezone,
        min_coverage_hours=args.min_coverage,
    )

    # Print summary statistics
    print(f"[aggregate] Aggregated to {len(daily_df)} daily records")
    if not daily_df.empty:
        date_range = f"{daily_df['date_local'].min().date()} to {daily_df['date_local'].max().date()}"
        print(f"[aggregate] Date range: {date_range}")

        avg_coverage = daily_df["coverage_hours"].mean()
        min_coverage = daily_df["coverage_hours"].min()
        max_coverage = daily_df["coverage_hours"].max()
        print(f"[aggregate] Coverage hours: min={min_coverage}, avg={avg_coverage:.1f}, max={max_coverage}")

        low_coverage_count = (daily_df["coverage_hours"] < args.min_coverage).sum()
        if low_coverage_count > 0:
            print(f"[aggregate] Days with low coverage (<{args.min_coverage}h): {low_coverage_count}")

    # Write output
    write_daily_tmax(daily_df, output_path)


if __name__ == "__main__":
    main()
