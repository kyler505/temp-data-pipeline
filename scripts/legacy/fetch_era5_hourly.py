#!/usr/bin/env python3
"""CLI for fetching ERA5 hourly 2m temperature data.

Usage:
    python scripts/fetch_era5_hourly.py --station KLGA --start 2010-01-01 --end 2015-12-31

For full options:
    python scripts/fetch_era5_hourly.py --help

Requirements:
    - cdsapi package installed
    - ~/.cdsapirc configured with CDS API credentials
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch ERA5 hourly 2m temperature data for a station.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch ERA5 data for KLGA for 2010
  python scripts/fetch_era5_hourly.py --station KLGA --start 2010-01-01 --end 2010-12-31

  # Force re-download (ignore cache)
  python scripts/fetch_era5_hourly.py --station KLGA --start 2010-01-01 --end 2010-12-31 --force
        """,
    )

    parser.add_argument(
        "--station",
        required=True,
        help="Station ID (e.g., KLGA)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD, exclusive)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for parquet files (default: data/raw/era5/<station>/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        from tempdata.fetch.era5_hourly import fetch_era5_hourly
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "\nTo use this script, install the required packages:\n"
            "  pip install cdsapi xarray netCDF4\n"
            "\nThen configure your CDS API credentials in ~/.cdsapirc",
            file=sys.stderr,
        )
        return 1

    try:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    except ValueError as e:
        print(f"Error parsing dates: {e}", file=sys.stderr)
        return 1

    print(f"[era5] Fetching ERA5 data for {args.station}")
    print(f"[era5] Date range: {start_date} to {end_date}")

    try:
        written_files = fetch_era5_hourly(
            station_id=args.station,
            start_date=start_date,
            end_date=end_date,
            out_dir=args.output_dir,
            force=args.force,
        )
    except Exception as e:
        print(f"Error fetching ERA5 data: {e}", file=sys.stderr)
        return 1

    print(f"\n[era5] Fetch complete. Wrote {len(written_files)} file(s):")
    for f in written_files:
        print(f"  - {f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
