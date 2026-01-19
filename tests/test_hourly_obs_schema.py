"""Tests for hourly_obs schema validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from tempdata.schemas.hourly_obs import (
    REQUIRED_COLUMNS,
    validate_hourly_obs,
)


class TestValidateHourlyObsPass:
    """Tests that should pass validation."""

    def test_fixture_validates(self, hourly_obs_sample: pd.DataFrame) -> None:
        """Sample fixture should pass validation."""
        # Should not raise
        validate_hourly_obs(hourly_obs_sample)

    def test_empty_dataframe_validates(self) -> None:
        """Empty DataFrame with correct columns should pass."""
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        validate_hourly_obs(df)

    def test_generated_data_validates(self, make_hourly_obs) -> None:
        """Factory-generated data should pass validation."""
        df = make_hourly_obs(n_rows=50)
        validate_hourly_obs(df)

    def test_with_null_temp_validates(self, make_hourly_obs) -> None:
        """DataFrame with null temp_c values should pass (missing data allowed)."""
        df = make_hourly_obs(n_rows=10)
        df.loc[5, "temp_c"] = None
        validate_hourly_obs(df)

    def test_edge_temperatures_validate(self, make_hourly_obs) -> None:
        """Edge case temperatures within range should pass."""
        df = make_hourly_obs(n_rows=4)
        df.loc[0, "temp_c"] = -90  # Minimum valid
        df.loc[1, "temp_c"] = 60  # Maximum valid
        df.loc[2, "temp_c"] = 0  # Zero
        df.loc[3, "temp_c"] = -40  # Polar cold
        validate_hourly_obs(df)

    def test_without_uniqueness_check(self, make_hourly_obs) -> None:
        """Duplicates allowed when require_unique_keys=False."""
        df = make_hourly_obs(n_rows=5)
        # Create a duplicate
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        # Should pass with uniqueness check disabled
        validate_hourly_obs(df, require_unique_keys=False)


class TestValidateHourlyObsFail:
    """Tests that should fail validation."""

    def test_missing_column_raises(self, make_hourly_obs) -> None:
        """Missing required column should raise ValueError."""
        df = make_hourly_obs(n_rows=5)
        df = df.drop(columns=["temp_c"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_hourly_obs(df)

    def test_null_ts_utc_raises(self, make_hourly_obs) -> None:
        """Null ts_utc should raise ValueError."""
        df = make_hourly_obs(n_rows=5)
        df.loc[2, "ts_utc"] = pd.NaT
        with pytest.raises(ValueError, match="Null values"):
            validate_hourly_obs(df)

    def test_null_station_id_raises(self, make_hourly_obs) -> None:
        """Null station_id should raise ValueError."""
        df = make_hourly_obs(n_rows=5)
        df.loc[2, "station_id"] = None
        with pytest.raises(ValueError, match="Null values"):
            validate_hourly_obs(df)

    def test_temp_below_range_raises(self, make_hourly_obs) -> None:
        """Temperature below -90 should raise ValueError."""
        df = make_hourly_obs(n_rows=5)
        df.loc[2, "temp_c"] = -100  # Below minimum
        with pytest.raises(ValueError, match="Out of range"):
            validate_hourly_obs(df)

    def test_temp_above_range_raises(self, make_hourly_obs) -> None:
        """Temperature above 60 should raise ValueError."""
        df = make_hourly_obs(n_rows=5)
        df.loc[2, "temp_c"] = 70  # Above maximum
        with pytest.raises(ValueError, match="Out of range"):
            validate_hourly_obs(df)

    def test_lat_out_of_range_raises(self, make_hourly_obs) -> None:
        """Latitude outside [-90, 90] should raise ValueError."""
        df = make_hourly_obs(n_rows=5)
        df.loc[2, "lat"] = 95  # Invalid latitude
        with pytest.raises(ValueError, match="Out of range"):
            validate_hourly_obs(df)

    def test_lon_out_of_range_raises(self, make_hourly_obs) -> None:
        """Longitude outside [-180, 180] should raise ValueError."""
        df = make_hourly_obs(n_rows=5)
        df.loc[2, "lon"] = -200  # Invalid longitude
        with pytest.raises(ValueError, match="Out of range"):
            validate_hourly_obs(df)

    def test_negative_qc_flags_raises(self, make_hourly_obs) -> None:
        """Negative qc_flags should raise ValueError."""
        df = make_hourly_obs(n_rows=5)
        df.loc[2, "qc_flags"] = -1
        with pytest.raises(ValueError, match="Negative values"):
            validate_hourly_obs(df)

    def test_duplicate_keys_raises(self, make_hourly_obs) -> None:
        """Duplicate (ts_utc, station_id) should raise with uniqueness enabled."""
        df = make_hourly_obs(n_rows=5)
        # Create a duplicate
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate keys"):
            validate_hourly_obs(df, require_unique_keys=True)

    def test_non_utc_timezone_raises(self) -> None:
        """Non-UTC timezone should raise ValueError."""
        # Create data with Eastern timezone instead of UTC
        timestamps = pd.date_range(
            start="2024-07-01",
            periods=5,
            freq="h",
            tz="America/New_York",
        )
        df = pd.DataFrame(
            {
                "ts_utc": timestamps,
                "station_id": "KLGA",
                "lat": 40.7769,
                "lon": -73.8740,
                "temp_c": [20, 21, 22, 23, 24],
                "source": "noaa",
                "qc_flags": 0,
            }
        )
        with pytest.raises(ValueError, match="(Wrong timezone|UTC)"):
            validate_hourly_obs(df)

    def test_naive_datetime_raises(self) -> None:
        """Timezone-naive datetime should raise ValueError."""
        # Create data without timezone
        timestamps = pd.date_range(
            start="2024-07-01",
            periods=5,
            freq="h",
        )  # No tz= parameter
        df = pd.DataFrame(
            {
                "ts_utc": timestamps,
                "station_id": "KLGA",
                "lat": 40.7769,
                "lon": -73.8740,
                "temp_c": [20, 21, 22, 23, 24],
                "source": "noaa",
                "qc_flags": 0,
            }
        )
        with pytest.raises(ValueError, match="Timezone required"):
            validate_hourly_obs(df)
