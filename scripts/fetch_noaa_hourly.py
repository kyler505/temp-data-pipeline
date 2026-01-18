"""CLI wrapper for NOAA hourly fetcher."""

from __future__ import annotations

import argparse

from tempdata.fetch.noaa_hourly import fetch_noaa_hourly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NOAA ISD hourly data.")
    parser.add_argument("--station", required=True, help="Station id, e.g. KLGA")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    parser.add_argument("--force", action="store_true", help="Re-download CSV cache")
    parser.add_argument("--no-cache", action="store_true", help="Do not keep CSV cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = fetch_noaa_hourly(
        station_id=args.station,
        start_date=args.start,
        end_date=args.end,
        force=args.force,
        use_cache=not args.no_cache,
    )
    print(f"[noaa] wrote {len(written)} files")
    for path in written:
        print(f"[noaa] {path}")


if __name__ == "__main__":
    main()
