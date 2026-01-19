"""Tests for daily Tmax aggregation logic."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from tempdata.aggregate.build_daily_tmax import build_daily_tmax, celsius_to_fahrenheit
from tempdata.schemas.daily_tmax import DAILY_TMAX_FIELDS, validate_daily_tmax
from tempdata.schemas.qc_flags import (
    QC_INCOMPLETE_DAY,
    QC_LOW_COVERAGE,
    QC_OK,
    QC_OUT_OF_RANGE,
    QC_SPIKE_DETECTED,
)


class TestCelsiusToFahrenheit:
    """Tests for temperature conversion."""

    def test_freezing_point(self) -> None:
        assert celsius_to_fahrenheit(0) == 32

    def test_boiling_point(self) -> None:
        assert celsius_to_fahrenheit(100) == 212

    def test_negative_40(self) -> None:
        """-40 is the same in both scales."""
        assert celsius_to_fahrenheit(-40) == -40

    def test_body_temp(self) -> None:
        # 37C = 98.6F
        assert abs(celsius_to_fahrenheit(37) - 98.6) < 0.1


class TestBuildDailyTmaxBasic:
    """Basic aggregation tests."""

    def test_empty_input_returns_empty_with_columns(self) -> None:
        """Empty input should return empty DataFrame with correct columns."""
        empty_df = pd.DataFrame(columns=[
            "ts_utc", "station_id", "lat", "lon", "temp_c", "source", "qc_flags"
        ])
        result = build_daily_tmax(empty_df, "America/New_York")

        assert result.empty
        assert list(result.columns) == DAILY_TMAX_FIELDS

    def test_basic_aggregation(self) -> None:
        """Basic aggregation produces correct Tmax."""
        # Create 24 hours of data for one local day
        # Start at 4 AM UTC = midnight Eastern (EDT is UTC-4 in July)
        start_ts = datetime(2024, 7, 1, 4, 0, 0, tzinfo=timezone.utc)
        timestamps = pd.date_range(start=start_ts, periods=24, freq="h", tz="UTC")

        df = pd.DataFrame({
            "ts_utc": timestamps,
            "station_id": "KLGA",
            "lat": 40.7769,
            "lon": -73.8740,
            "temp_c": [20.0 + (i % 10) for i in range(24)],
            "source": "noaa",
            "qc_flags": 0,
        })

        result = build_daily_tmax(df, "America/New_York")

        # Should have 1 day (all hours map to July 1 local)
        assert len(result) == 1

        # Tmax should be max of temp_base + (i % 10) = 20 + 9 = 29
        assert result.iloc[0]["tmax_c"] == 29.0
        assert result.iloc[0]["coverage_hours"] == 24

    def test_validates_output_schema(self, make_hourly_obs) -> None:
        """Output should pass schema validation."""
        df = make_hourly_obs(n_rows=48)
        result = build_daily_tmax(df, "America/New_York")

        # Should not raise
        validate_daily_tmax(result)

    def test_source_is_noaa_isd(self, make_hourly_obs) -> None:
        """Source field should be 'noaa_isd'."""
        df = make_hourly_obs(n_rows=24)
        result = build_daily_tmax(df, "America/New_York")

        assert result.iloc[0]["source"] == "noaa_isd"


class TestTimezoneConversion:
    """Tests for timezone handling."""

    def test_groups_by_local_date_not_utc(self) -> None:
        """Tmax should be computed per local calendar day, not UTC day."""
        # Create observations that span midnight UTC but are same day in NY
        # UTC 2024-07-01 23:00 = NY 2024-07-01 19:00 (EDT, UTC-4)
        # UTC 2024-07-02 03:00 = NY 2024-07-01 23:00 (EDT, UTC-4)
        # UTC 2024-07-02 05:00 = NY 2024-07-02 01:00 (EDT, UTC-4)
        timestamps = [
            pd.Timestamp("2024-07-01 23:00:00", tz="UTC"),  # NY: July 1, 7pm
            pd.Timestamp("2024-07-02 03:00:00", tz="UTC"),  # NY: July 1, 11pm
            pd.Timestamp("2024-07-02 05:00:00", tz="UTC"),  # NY: July 2, 1am
        ]

        df = pd.DataFrame({
            "ts_utc": timestamps,
            "station_id": "KLGA",
            "lat": 40.7769,
            "lon": -73.8740,
            "temp_c": [30.0, 35.0, 25.0],  # Max on July 1 (local) should be 35
            "source": "noaa",
            "qc_flags": QC_OK,
        })

        result = build_daily_tmax(df, "America/New_York")

        # Should have 2 days
        assert len(result) == 2

        # July 1 (local) should have max 35 (from the 3am UTC observation)
        july_1 = result[result["date_local"].dt.day == 1]
        assert len(july_1) == 1
        assert july_1.iloc[0]["tmax_c"] == 35.0
        assert july_1.iloc[0]["coverage_hours"] == 2

        # July 2 (local) should have max 25
        july_2 = result[result["date_local"].dt.day == 2]
        assert len(july_2) == 1
        assert july_2.iloc[0]["tmax_c"] == 25.0
        assert july_2.iloc[0]["coverage_hours"] == 1

    def test_date_local_is_midnight(self, make_hourly_obs) -> None:
        """date_local should be at midnight (no time component)."""
        df = make_hourly_obs(n_rows=48)
        result = build_daily_tmax(df, "America/New_York")

        for date_val in result["date_local"]:
            # After removing tz, should be at midnight
            naive_dt = date_val.tz_localize(None) if date_val.tz else date_val
            assert naive_dt.hour == 0
            assert naive_dt.minute == 0
            assert naive_dt.second == 0


class TestQCExclusion:
    """Tests for QC-based hour exclusion."""

    def test_out_of_range_excluded_from_tmax(self) -> None:
        """Hours with QC_OUT_OF_RANGE should not contribute to Tmax."""
        timestamps = pd.date_range(
            start="2024-07-01 04:00:00",  # Start at midnight EDT (4am UTC)
            periods=4,
            freq="h",
            tz="UTC",
        )

        df = pd.DataFrame({
            "ts_utc": timestamps,
            "station_id": "KLGA",
            "lat": 40.7769,
            "lon": -73.8740,
            "temp_c": [30.0, 99.0, 25.0, 28.0],  # 99 is flagged out of range
            "source": "noaa",
            "qc_flags": [QC_OK, QC_OUT_OF_RANGE, QC_OK, QC_OK],
        })

        result = build_daily_tmax(df, "America/New_York")

        assert len(result) == 1
        # Tmax should be 30, not 99 (which is flagged)
        assert result.iloc[0]["tmax_c"] == 30.0
        # Coverage should be 3 (excluding the flagged hour)
        assert result.iloc[0]["coverage_hours"] == 3

    def test_spike_flagged_included_in_tmax(self) -> None:
        """Hours with QC_SPIKE_DETECTED should still contribute to Tmax."""
        timestamps = pd.date_range(
            start="2024-07-01 04:00:00",
            periods=4,
            freq="h",
            tz="UTC",
        )

        df = pd.DataFrame({
            "ts_utc": timestamps,
            "station_id": "KLGA",
            "lat": 40.7769,
            "lon": -73.8740,
            "temp_c": [30.0, 45.0, 25.0, 28.0],  # 45 is spike but real heat
            "source": "noaa",
            "qc_flags": [QC_OK, QC_SPIKE_DETECTED, QC_OK, QC_OK],
        })

        result = build_daily_tmax(df, "America/New_York")

        # Tmax SHOULD be 45 (spike-flagged values are included per design doc)
        assert result.iloc[0]["tmax_c"] == 45.0
        assert result.iloc[0]["coverage_hours"] == 4

    def test_null_temp_excluded(self) -> None:
        """Hours with null temp_c should not contribute to Tmax."""
        timestamps = pd.date_range(
            start="2024-07-01 04:00:00",
            periods=4,
            freq="h",
            tz="UTC",
        )

        df = pd.DataFrame({
            "ts_utc": timestamps,
            "station_id": "KLGA",
            "lat": 40.7769,
            "lon": -73.8740,
            "temp_c": [30.0, None, 25.0, 28.0],
            "source": "noaa",
            "qc_flags": QC_OK,
        })

        result = build_daily_tmax(df, "America/New_York")

        assert result.iloc[0]["tmax_c"] == 30.0
        assert result.iloc[0]["coverage_hours"] == 3


class TestQCFlagPropagation:
    """Tests for QC flag propagation to daily level."""

    def test_spike_flag_propagates(self) -> None:
        """QC_SPIKE_DETECTED should propagate to daily level."""
        timestamps = pd.date_range(
            start="2024-07-01 04:00:00",
            periods=4,
            freq="h",
            tz="UTC",
        )

        df = pd.DataFrame({
            "ts_utc": timestamps,
            "station_id": "KLGA",
            "lat": 40.7769,
            "lon": -73.8740,
            "temp_c": [30.0, 35.0, 25.0, 28.0],
            "source": "noaa",
            "qc_flags": [QC_OK, QC_SPIKE_DETECTED, QC_OK, QC_OK],
        })

        result = build_daily_tmax(df, "America/New_York")

        # Daily QC flags should include the spike flag
        assert (result.iloc[0]["qc_flags"] & QC_SPIKE_DETECTED) != 0

    def test_multiple_flags_combined_with_or(self) -> None:
        """Multiple hourly flags should be combined with bitwise OR."""
        timestamps = pd.date_range(
            start="2024-07-01 04:00:00",
            periods=4,
            freq="h",
            tz="UTC",
        )

        df = pd.DataFrame({
            "ts_utc": timestamps,
            "station_id": "KLGA",
            "lat": 40.7769,
            "lon": -73.8740,
            "temp_c": [30.0, 35.0, 25.0, 28.0],
            "source": "noaa",
            "qc_flags": [QC_SPIKE_DETECTED, QC_OUT_OF_RANGE, QC_OK, QC_OK],
        })

        result = build_daily_tmax(df, "America/New_York")

        # Both flags should be present in daily
        assert (result.iloc[0]["qc_flags"] & QC_SPIKE_DETECTED) != 0
        assert (result.iloc[0]["qc_flags"] & QC_OUT_OF_RANGE) != 0


class TestCoverageFlags:
    """Tests for coverage-based QC flags."""

    def test_low_coverage_flag_applied(self) -> None:
        """QC_LOW_COVERAGE should be set when coverage < threshold."""
        # Create only 10 hours of data
        timestamps = pd.date_range(
            start="2024-07-01 04:00:00",
            periods=10,
            freq="h",
            tz="UTC",
        )

        df = pd.DataFrame({
            "ts_utc": timestamps,
            "station_id": "KLGA",
            "lat": 40.7769,
            "lon": -73.8740,
            "temp_c": [25.0 + i for i in range(10)],
            "source": "noaa",
            "qc_flags": QC_OK,
        })

        result = build_daily_tmax(df, "America/New_York", min_coverage_hours=18)

        assert (result.iloc[0]["qc_flags"] & QC_LOW_COVERAGE) != 0
        assert result.iloc[0]["coverage_hours"] == 10

    def test_incomplete_day_flag_when_no_valid_hours(self) -> None:
        """QC_INCOMPLETE_DAY should be set when coverage == 0."""
        # All hours are out of range
        timestamps = pd.date_range(
            start="2024-07-01 04:00:00",
            periods=4,
            freq="h",
            tz="UTC",
        )

        df = pd.DataFrame({
            "ts_utc": timestamps,
            "station_id": "KLGA",
            "lat": 40.7769,
            "lon": -73.8740,
            "temp_c": [99.0, 99.0, 99.0, 99.0],  # All flagged
            "source": "noaa",
            "qc_flags": QC_OUT_OF_RANGE,
        })

        result = build_daily_tmax(df, "America/New_York")

        assert result.iloc[0]["coverage_hours"] == 0
        assert (result.iloc[0]["qc_flags"] & QC_INCOMPLETE_DAY) != 0

    def test_good_coverage_no_flag(self) -> None:
        """Days with sufficient coverage should not have low coverage flag."""
        timestamps = pd.date_range(
            start="2024-07-01 04:00:00",
            periods=20,
            freq="h",
            tz="UTC",
        )

        df = pd.DataFrame({
            "ts_utc": timestamps,
            "station_id": "KLGA",
            "lat": 40.7769,
            "lon": -73.8740,
            "temp_c": [25.0 + (i % 10) for i in range(20)],
            "source": "noaa",
            "qc_flags": QC_OK,
        })

        result = build_daily_tmax(df, "America/New_York", min_coverage_hours=18)

        assert (result.iloc[0]["qc_flags"] & QC_LOW_COVERAGE) == 0
        assert (result.iloc[0]["qc_flags"] & QC_INCOMPLETE_DAY) == 0


class TestFahrenheitConversion:
    """Tests for Celsius to Fahrenheit conversion in output."""

    def test_fahrenheit_matches_celsius(self, make_hourly_obs) -> None:
        """tmax_f should match tmax_c conversion."""
        df = make_hourly_obs(n_rows=24, temp_base=30.0)
        result = build_daily_tmax(df, "America/New_York")

        tmax_c = result.iloc[0]["tmax_c"]
        tmax_f = result.iloc[0]["tmax_f"]
        expected_f = round(tmax_c * 9 / 5 + 32, 1)

        assert abs(tmax_f - expected_f) < 0.2


class TestMultipleStations:
    """Tests for handling multiple stations."""

    def test_groups_by_station(self) -> None:
        """Should aggregate separately for each station."""
        timestamps = pd.date_range(
            start="2024-07-01 04:00:00",
            periods=4,
            freq="h",
            tz="UTC",
        )

        df = pd.DataFrame({
            "ts_utc": list(timestamps) * 2,
            "station_id": ["KLGA"] * 4 + ["KJFK"] * 4,
            "lat": [40.7769] * 4 + [40.6413] * 4,
            "lon": [-73.8740] * 4 + [-73.7781] * 4,
            "temp_c": [30.0, 35.0, 25.0, 28.0, 32.0, 38.0, 26.0, 29.0],
            "source": "noaa",
            "qc_flags": QC_OK,
        })

        result = build_daily_tmax(df, "America/New_York")

        assert len(result) == 2

        klga = result[result["station_id"] == "KLGA"]
        kjfk = result[result["station_id"] == "KJFK"]

        assert klga.iloc[0]["tmax_c"] == 35.0
        assert kjfk.iloc[0]["tmax_c"] == 38.0


class TestUpdatedAtUtc:
    """Tests for updated_at_utc field."""

    def test_updated_at_is_utc(self, make_hourly_obs) -> None:
        """updated_at_utc should be timezone-aware UTC."""
        df = make_hourly_obs(n_rows=24)
        result = build_daily_tmax(df, "America/New_York")

        updated_at = result.iloc[0]["updated_at_utc"]
        assert updated_at.tz is not None
        assert str(updated_at.tz).upper() in ("UTC", "TIMEZONE.UTC")

    def test_updated_at_is_recent(self, make_hourly_obs) -> None:
        """updated_at_utc should be close to current time."""
        df = make_hourly_obs(n_rows=24)
        before = pd.Timestamp.now(tz=timezone.utc)
        result = build_daily_tmax(df, "America/New_York")
        after = pd.Timestamp.now(tz=timezone.utc)

        updated_at = result.iloc[0]["updated_at_utc"]
        assert before <= updated_at <= after


class TestSubHourlyData:
    """Tests for handling sub-hourly data (multiple observations per hour)."""

    def test_coverage_counts_unique_hours_not_observations(self) -> None:
        """coverage_hours should count unique hours, not total observations.

        NOAA ISD data often has multiple observations per hour (e.g., 00:00, 00:51).
        We should count unique hours with valid data, capped at 24.
        """
        # Create sub-hourly data: 3 observations in hour 0, 2 in hour 1, 1 in hour 2
        timestamps = [
            pd.Timestamp("2024-07-01 04:00:00", tz="UTC"),  # hour 0 local
            pd.Timestamp("2024-07-01 04:20:00", tz="UTC"),  # hour 0 local (duplicate)
            pd.Timestamp("2024-07-01 04:51:00", tz="UTC"),  # hour 0 local (duplicate)
            pd.Timestamp("2024-07-01 05:00:00", tz="UTC"),  # hour 1 local
            pd.Timestamp("2024-07-01 05:30:00", tz="UTC"),  # hour 1 local (duplicate)
            pd.Timestamp("2024-07-01 06:00:00", tz="UTC"),  # hour 2 local
        ]

        df = pd.DataFrame({
            "ts_utc": timestamps,
            "station_id": "KLGA",
            "lat": 40.7769,
            "lon": -73.8740,
            "temp_c": [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
            "source": "noaa",
            "qc_flags": QC_OK,
        })

        result = build_daily_tmax(df, "America/New_York")

        # Should count 3 unique hours, not 6 observations
        assert result.iloc[0]["coverage_hours"] == 3
        # Tmax should still be the max of all valid observations
        assert result.iloc[0]["tmax_c"] == 30.0

    def test_sub_hourly_coverage_capped_at_24(self) -> None:
        """Even with many sub-hourly observations, coverage_hours <= 24."""
        # Create 48 observations across 24 hours (2 per hour)
        base = pd.Timestamp("2024-07-01 04:00:00", tz="UTC")
        timestamps = []
        for hour in range(24):
            timestamps.append(base + pd.Timedelta(hours=hour))
            timestamps.append(base + pd.Timedelta(hours=hour, minutes=30))

        df = pd.DataFrame({
            "ts_utc": timestamps,
            "station_id": "KLGA",
            "lat": 40.7769,
            "lon": -73.8740,
            "temp_c": [20.0 + (i % 10) for i in range(48)],
            "source": "noaa",
            "qc_flags": QC_OK,
        })

        result = build_daily_tmax(df, "America/New_York")

        # Should count exactly 24 unique hours
        assert result.iloc[0]["coverage_hours"] == 24
        # Schema validation should pass
        validate_daily_tmax(result)
