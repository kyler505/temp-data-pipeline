"""Main script to run the data pipeline.

Pipeline flow:
    fetch_noaa_hourly -> clean_hourly_file -> (aggregate: future)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tempdata.clean import clean_hourly_file
from tempdata.fetch.noaa_hourly import fetch_noaa_hourly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temp data pipeline.")
    parser.add_argument("--station", required=True, help="Station id, e.g. KLGA")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (default: data)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    # Stage 1: Fetch raw data
    raw_dir = data_dir / "raw" / "noaa_hourly" / args.station
    cache_dir = data_dir / "cache" / "isd_csv" / args.station

    print(f"[pipeline] Fetching data for {args.station} from {args.start} to {args.end}")
    written = fetch_noaa_hourly(
        station_id=args.station,
        start_date=args.start,
        end_date=args.end,
        out_dir=raw_dir,
        cache_dir=cache_dir,
    )
    print(f"[pipeline] Fetched {len(written)} raw parquet files")

    # Stage 2: Clean each raw file
    clean_dir = data_dir / "clean" / "noaa_hourly" / args.station
    cleaned_files = []

    for raw_path in written:
        clean_path = clean_dir / raw_path.name
        print(f"\n[pipeline] Cleaning {raw_path.name}")
        clean_hourly_file(raw_path, clean_path)
        cleaned_files.append(clean_path)

    print(f"\n[pipeline] Cleaned {len(cleaned_files)} files")

    # Stage 3: Aggregate to daily Tmax (future)
    # TODO: Implement daily Tmax aggregation


if __name__ == "__main__":
    main()
