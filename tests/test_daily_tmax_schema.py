"""Tests for daily_tmax schema validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from tempdata.schemas.daily_tmax import (
    REQUIRED_COLUMNS,
    validate_daily_tmax,
)


class TestValidateDailyTmaxPass:
    """Tests that should pass validation."""

    def test_fixture_validates(self, daily_tmax_sample: pd.DataFrame) -> None:
        """Sample fixture should pass validation."""
        # Should not raise
        validate_daily_tmax(daily_tmax_sample)

    def test_empty_dataframe_validates(self) -> None:
        """Empty DataFrame with correct columns should pass."""
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        validate_daily_tmax(df)

    def test_generated_data_validates(self, make_daily_tmax) -> None:
        """Factory-generated data should pass validation."""
        df = make_daily_tmax(n_rows=10)
        validate_daily_tmax(df)

    def test_edge_temperatures_validate(self, make_daily_tmax) -> None:
        """Edge case temperatures within range should pass."""
        df = make_daily_tmax(n_rows=4)
        # Set edge values with correct C/F conversion
        df.loc[0, "tmax_c"] = -90
        df.loc[0, "tmax_f"] = round(-90 * 9 / 5 + 32, 1)  # -130
        df.loc[1, "tmax_c"] = 60
        df.loc[1, "tmax_f"] = round(60 * 9 / 5 + 32, 1)  # 140
        df.loc[2, "tmax_c"] = 0
        df.loc[2, "tmax_f"] = 32.0
        df.loc[3, "tmax_c"] = -40
        df.loc[3, "tmax_f"] = -40.0  # -40 is same in C and F
        validate_daily_tmax(df)

    def test_zero_coverage_validates(self, make_daily_tmax) -> None:
        """Zero coverage hours should pass (edge case)."""
        df = make_daily_tmax(n_rows=3)
        df.loc[1, "coverage_hours"] = 0
        validate_daily_tmax(df)


class TestValidateDailyTmaxFail:
    """Tests that should fail validation."""

    def test_missing_column_raises(self, make_daily_tmax) -> None:
        """Missing required column should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        df = df.drop(columns=["tmax_c"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_daily_tmax(df)

    def test_null_tmax_c_raises(self, make_daily_tmax) -> None:
        """Null tmax_c should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        df.loc[2, "tmax_c"] = None
        with pytest.raises(ValueError, match="Null values"):
            validate_daily_tmax(df)

    def test_duplicate_date_station_raises(self, make_daily_tmax) -> None:
        """Duplicate (date_local, station_id) should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        # Create a duplicate
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate keys"):
            validate_daily_tmax(df)

    def test_coverage_hours_above_24_raises(self, make_daily_tmax) -> None:
        """Coverage hours > 24 should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        df.loc[2, "coverage_hours"] = 25
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax(df)

    def test_coverage_hours_negative_raises(self, make_daily_tmax) -> None:
        """Negative coverage hours should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        df.loc[2, "coverage_hours"] = -1
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax(df)

    def test_tmax_c_below_range_raises(self, make_daily_tmax) -> None:
        """tmax_c below -90 should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        df.loc[2, "tmax_c"] = -100
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax(df)

    def test_tmax_c_above_range_raises(self, make_daily_tmax) -> None:
        """tmax_c above 60 should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        df.loc[2, "tmax_c"] = 70
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax(df)

    def test_tmax_f_below_range_raises(self, make_daily_tmax) -> None:
        """tmax_f below -130 should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        df.loc[2, "tmax_f"] = -140
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax(df)

    def test_tmax_f_above_range_raises(self, make_daily_tmax) -> None:
        """tmax_f above 140 should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        df.loc[2, "tmax_f"] = 150
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax(df)

    def test_negative_qc_flags_raises(self, make_daily_tmax) -> None:
        """Negative qc_flags should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        df.loc[2, "qc_flags"] = -1
        with pytest.raises(ValueError, match="Negative values"):
            validate_daily_tmax(df)

    def test_cf_inconsistency_raises(self, make_daily_tmax) -> None:
        """Celsius/Fahrenheit inconsistency should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        # Set tmax_f to wrong value (should be 86 for 30C, but we set 100)
        df.loc[2, "tmax_c"] = 30
        df.loc[2, "tmax_f"] = 100  # Wrong! Should be 86
        with pytest.raises(ValueError, match="Values not close"):
            validate_daily_tmax(df)

    def test_non_utc_updated_at_raises(self, make_daily_tmax) -> None:
        """Non-UTC updated_at_utc should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        # Set to Eastern timezone
        df["updated_at_utc"] = pd.Timestamp.now(tz="America/New_York")
        with pytest.raises(ValueError, match="(Wrong timezone|UTC)"):
            validate_daily_tmax(df)

    def test_naive_updated_at_raises(self, make_daily_tmax) -> None:
        """Timezone-naive updated_at_utc should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        # Set to naive datetime
        df["updated_at_utc"] = pd.Timestamp.now()  # No timezone
        with pytest.raises(ValueError, match="Timezone required"):
            validate_daily_tmax(df)

    def test_date_with_time_component_raises(self, make_daily_tmax) -> None:
        """date_local with non-midnight time should raise ValueError."""
        df = make_daily_tmax(n_rows=5)
        # Add a time component
        df.loc[2, "date_local"] = pd.Timestamp("2024-07-03 14:30:00")
        with pytest.raises(ValueError, match="Date has time component"):
            validate_daily_tmax(df)
