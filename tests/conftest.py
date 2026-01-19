"""Pytest configuration and fixtures."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

# Directory containing test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def hourly_obs_sample() -> pd.DataFrame:
    """Load sample hourly observations fixture."""
    path = FIXTURES_DIR / "hourly_obs_sample.parquet"
    return pd.read_parquet(path)


@pytest.fixture
def daily_tmax_sample() -> pd.DataFrame:
    """Load sample daily tmax fixture."""
    path = FIXTURES_DIR / "daily_tmax_sample.parquet"
    return pd.read_parquet(path)


@pytest.fixture
def make_hourly_obs():
    """Factory fixture for creating hourly observation DataFrames."""

    def _make(
        n_rows: int = 10,
        station_id: str = "KLGA",
        start_ts: datetime | None = None,
        temp_base: float = 20.0,
    ) -> pd.DataFrame:
        if start_ts is None:
            start_ts = datetime(2024, 7, 1, 0, 0, 0, tzinfo=timezone.utc)

        timestamps = pd.date_range(
            start=start_ts,
            periods=n_rows,
            freq="h",
            tz="UTC",
        )

        return pd.DataFrame(
            {
                "ts_utc": timestamps,
                "station_id": station_id,
                "lat": 40.7769,
                "lon": -73.8740,
                "temp_c": [temp_base + (i % 10) for i in range(n_rows)],
                "source": "noaa",
                "qc_flags": 0,
            }
        )

    return _make


@pytest.fixture
def make_daily_tmax():
    """Factory fixture for creating daily tmax DataFrames."""

    def _make(
        n_rows: int = 5,
        station_id: str = "KLGA",
        start_date: datetime | None = None,
        tmax_base: float = 30.0,
    ) -> pd.DataFrame:
        if start_date is None:
            start_date = datetime(2024, 7, 1, 0, 0, 0)

        dates = pd.date_range(
            start=start_date,
            periods=n_rows,
            freq="D",
        )

        tmax_c_values = [tmax_base + (i % 5) for i in range(n_rows)]
        tmax_f_values = [round(c * 9 / 5 + 32, 1) for c in tmax_c_values]

        return pd.DataFrame(
            {
                "date_local": dates,
                "station_id": station_id,
                "tmax_c": tmax_c_values,
                "tmax_f": tmax_f_values,
                "coverage_hours": 24,
                "source": "noaa",
                "qc_flags": 0,
                "updated_at_utc": pd.Timestamp.now(tz=timezone.utc),
            }
        )

    return _make
