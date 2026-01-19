#!/usr/bin/env python3
"""Generate test fixture parquet files.

Run this script to create/regenerate the fixture files:
    python tests/generate_fixtures.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def generate_hourly_obs_sample() -> None:
    """Generate sample hourly observations fixture.

    Creates a small dataset with:
    - Normal temperature values
    - A few edge cases (near-zero, negative)
    - At least one QC-flagged record
    """
    # Create 100 rows spanning a few days
    start_ts = datetime(2024, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = pd.date_range(start=start_ts, periods=100, freq="h", tz="UTC")

    # Temperature pattern: varies from 15-35C with some variation
    temps = [20 + 10 * ((i % 24) / 24) + (i % 5) - 2 for i in range(100)]

    df = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "station_id": "KLGA",
            "lat": 40.7769,
            "lon": -73.8740,
            "temp_c": temps,
            "source": "noaa",
            "qc_flags": 0,
        }
    )

    # Add one QC-flagged record (still schema-valid, just flagged)
    df.loc[50, "qc_flags"] = 1  # QC_MISSING_VALUE flag

    # Ensure proper dtypes
    df["qc_flags"] = df["qc_flags"].astype(int)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIXTURES_DIR / "hourly_obs_sample.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Generated {output_path} with {len(df)} rows")


def generate_daily_tmax_sample() -> None:
    """Generate sample daily tmax fixture.

    Creates a small dataset with:
    - Normal daily max temperatures
    - Consistent C/F values
    - At least one QC-flagged record
    """
    # Create 20 rows spanning 20 days
    start_date = datetime(2024, 7, 1, 0, 0, 0)
    dates = pd.date_range(start=start_date, periods=20, freq="D")

    # Temperature pattern: summer temps 28-36C
    tmax_c = [30 + (i % 7) - 2 for i in range(20)]
    tmax_f = [round(c * 9 / 5 + 32, 1) for c in tmax_c]

    df = pd.DataFrame(
        {
            "date_local": dates,
            "station_id": "KLGA",
            "tmax_c": tmax_c,
            "tmax_f": tmax_f,
            "coverage_hours": 24,
            "source": "noaa",
            "qc_flags": 0,
            "updated_at_utc": pd.Timestamp.now(tz=timezone.utc),
        }
    )

    # Add one low-coverage day with QC flag
    df.loc[10, "coverage_hours"] = 18
    df.loc[10, "qc_flags"] = 16  # QC_LOW_COVERAGE flag

    # Ensure proper dtypes
    df["qc_flags"] = df["qc_flags"].astype(int)
    df["coverage_hours"] = df["coverage_hours"].astype(int)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIXTURES_DIR / "daily_tmax_sample.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Generated {output_path} with {len(df)} rows")


if __name__ == "__main__":
    generate_hourly_obs_sample()
    generate_daily_tmax_sample()
    print("Done generating fixtures.")
