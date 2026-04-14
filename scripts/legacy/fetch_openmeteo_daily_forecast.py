"""CLI wrapper for Open-Meteo daily Tmax forecast fetcher."""

from __future__ import annotations

import argparse

import pandas as pd

from tempdata.fetch.noaa_hourly import resolve_station
from tempdata.fetch.openmeteo_daily_forecast import fetch_openmeteo_daily_tmax_forecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch daily Tmax forecast from Open-Meteo."
    )
    parser.add_argument(
        "--station",
        required=True,
        help="Station id, e.g. KLGA",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=14,
        help="Number of days to forecast (default: 14)",
    )
    parser.add_argument(
        "--write-raw",
        action="store_true",
        help="Write raw JSON response",
    )
    parser.add_argument(
        "--out-raw",
        type=str,
        default=None,
        help="Output directory for raw JSON (optional)",
    )
    parser.add_argument(
        "--out-parquet",
        type=str,
        default=None,
        help="Output directory for parquet files (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve station to get metadata
    station = resolve_station(args.station)

    print(f"[openmeteo] Fetching forecast for {station.station_id}")
    print(f"[openmeteo] Location: ({station.lat}, {station.lon})")
    print(f"[openmeteo] Timezone: {station.tz}")
    print(f"[openmeteo] Forecast days: {args.forecast_days}")

    written = fetch_openmeteo_daily_tmax_forecast(
        station_id=station.station_id,
        lat=station.lat,
        lon=station.lon,
        station_tz=station.tz,
        out_raw_dir=args.out_raw,
        out_parquet_dir=args.out_parquet,
        forecast_days=args.forecast_days,
        write_raw=args.write_raw,
    )

    print(f"[openmeteo] Wrote {len(written)} files:")
    for path in written:
        print(f"[openmeteo]   {path}")

    # Load parquet and print summary
    parquet_files = [p for p in written if p.suffix == ".parquet"]
    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
        if not df.empty:
            print(f"\n[openmeteo] Summary:")
            print(f"[openmeteo]   Issue time: {df['issue_time_utc'].iloc[0]}")
            print(f"[openmeteo]   Rows: {len(df)}")
            print(f"[openmeteo]   Target dates: {df['target_date_local'].min()} to {df['target_date_local'].max()}")
            print(f"[openmeteo]   Tmax (C): {df['tmax_pred_c'].min():.1f} to {df['tmax_pred_c'].max():.1f}")
            print(f"[openmeteo]   Tmax (F): {df['tmax_pred_f'].min():.1f} to {df['tmax_pred_f'].max():.1f}")
            print(f"[openmeteo]   Lead hours: {df['lead_hours'].min()} to {df['lead_hours'].max()}")


if __name__ == "__main__":
    main()
