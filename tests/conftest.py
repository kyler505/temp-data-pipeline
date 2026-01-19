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


@pytest.fixture
def make_daily_tmax_forecast():
    """Factory fixture for creating daily tmax forecast DataFrames."""

    def _make(
        n_rows: int = 7,
        station_id: str = "KLGA",
        lat: float = 40.7769,
        lon: float = -73.8740,
        issue_time: datetime | None = None,
        start_date: datetime | None = None,
        tmax_base: float = 25.0,
    ) -> pd.DataFrame:
        if issue_time is None:
            issue_time = datetime(2024, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

        if start_date is None:
            start_date = datetime(2024, 7, 1, 0, 0, 0)

        target_dates = pd.date_range(
            start=start_date,
            periods=n_rows,
            freq="D",
        )

        tmax_c_values = [tmax_base + (i % 10) for i in range(n_rows)]
        tmax_f_values = [round(c * 9 / 5 + 32, 1) for c in tmax_c_values]
        # Lead hours: hours from issue time to target date midnight
        # For simplicity, assume target date midnight is in UTC-5 (EST)
        lead_hours_values = [12 + (i * 24) for i in range(n_rows)]

        return pd.DataFrame(
            {
                "station_id": station_id,
                "lat": lat,
                "lon": lon,
                "issue_time_utc": pd.Timestamp(issue_time),
                "target_date_local": target_dates,
                "tmax_pred_c": tmax_c_values,
                "tmax_pred_f": tmax_f_values,
                "lead_hours": lead_hours_values,
                "model": "openmeteo",
                "source": "openmeteo",
                "ingested_at_utc": pd.Timestamp(issue_time),
            }
        )

    return _make
