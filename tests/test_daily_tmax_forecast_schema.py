"""Tests for daily_tmax_forecast schema validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from tempdata.schemas.daily_tmax_forecast import (
    REQUIRED_COLUMNS,
    validate_daily_tmax_forecast,
)


class TestValidateDailyTmaxForecastPass:
    """Tests that should pass validation."""

    def test_empty_dataframe_validates(self) -> None:
        """Empty DataFrame with correct columns should pass."""
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        validate_daily_tmax_forecast(df)

    def test_generated_data_validates(self, make_daily_tmax_forecast) -> None:
        """Factory-generated data should pass validation."""
        df = make_daily_tmax_forecast(n_rows=10)
        validate_daily_tmax_forecast(df)

    def test_edge_temperatures_validate(self, make_daily_tmax_forecast) -> None:
        """Edge case temperatures within range should pass."""
        df = make_daily_tmax_forecast(n_rows=4)
        # Set edge values with correct C/F conversion
        df.loc[0, "tmax_pred_c"] = -90
        df.loc[0, "tmax_pred_f"] = round(-90 * 9 / 5 + 32, 1)  # -130
        df.loc[1, "tmax_pred_c"] = 60
        df.loc[1, "tmax_pred_f"] = round(60 * 9 / 5 + 32, 1)  # 140
        df.loc[2, "tmax_pred_c"] = 0
        df.loc[2, "tmax_pred_f"] = 32.0
        df.loc[3, "tmax_pred_c"] = -40
        df.loc[3, "tmax_pred_f"] = -40.0  # -40 is same in C and F
        validate_daily_tmax_forecast(df)

    def test_edge_lead_hours_validate(self, make_daily_tmax_forecast) -> None:
        """Edge case lead hours within range should pass."""
        df = make_daily_tmax_forecast(n_rows=3)
        df.loc[0, "lead_hours"] = -24  # Minimum
        df.loc[1, "lead_hours"] = 0
        df.loc[2, "lead_hours"] = 720  # Maximum (30 days)
        validate_daily_tmax_forecast(df)


class TestValidateDailyTmaxForecastFail:
    """Tests that should fail validation."""

    def test_missing_column_raises(self, make_daily_tmax_forecast) -> None:
        """Missing required column should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        df = df.drop(columns=["tmax_pred_c"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_daily_tmax_forecast(df)

    def test_null_tmax_pred_c_raises(self, make_daily_tmax_forecast) -> None:
        """Null tmax_pred_c should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        df.loc[2, "tmax_pred_c"] = None
        with pytest.raises(ValueError, match="Null values"):
            validate_daily_tmax_forecast(df)

    def test_duplicate_keys_raises(self, make_daily_tmax_forecast) -> None:
        """Duplicate (station_id, issue_time_utc, target_date_local) should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        # Create a duplicate
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate keys"):
            validate_daily_tmax_forecast(df)

    def test_lead_hours_below_range_raises(self, make_daily_tmax_forecast) -> None:
        """Lead hours < -24 should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        df.loc[2, "lead_hours"] = -48
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax_forecast(df)

    def test_lead_hours_above_range_raises(self, make_daily_tmax_forecast) -> None:
        """Lead hours > 720 should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        df.loc[2, "lead_hours"] = 800
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax_forecast(df)

    def test_tmax_pred_c_below_range_raises(self, make_daily_tmax_forecast) -> None:
        """tmax_pred_c below -90 should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        df.loc[2, "tmax_pred_c"] = -100
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax_forecast(df)

    def test_tmax_pred_c_above_range_raises(self, make_daily_tmax_forecast) -> None:
        """tmax_pred_c above 60 should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        df.loc[2, "tmax_pred_c"] = 70
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax_forecast(df)

    def test_tmax_pred_f_below_range_raises(self, make_daily_tmax_forecast) -> None:
        """tmax_pred_f below -130 should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        df.loc[2, "tmax_pred_f"] = -140
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax_forecast(df)

    def test_tmax_pred_f_above_range_raises(self, make_daily_tmax_forecast) -> None:
        """tmax_pred_f above 140 should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        df.loc[2, "tmax_pred_f"] = 150
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax_forecast(df)

    def test_lat_out_of_range_raises(self, make_daily_tmax_forecast) -> None:
        """Latitude outside [-90, 90] should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        df.loc[2, "lat"] = 100
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax_forecast(df)

    def test_lon_out_of_range_raises(self, make_daily_tmax_forecast) -> None:
        """Longitude outside [-180, 180] should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        df.loc[2, "lon"] = 200
        with pytest.raises(ValueError, match="Out of range"):
            validate_daily_tmax_forecast(df)

    def test_cf_inconsistency_raises(self, make_daily_tmax_forecast) -> None:
        """Celsius/Fahrenheit inconsistency should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        # Set tmax_pred_f to wrong value (should be 86 for 30C, but we set 100)
        df.loc[2, "tmax_pred_c"] = 30
        df.loc[2, "tmax_pred_f"] = 100  # Wrong! Should be 86
        with pytest.raises(ValueError, match="Values not close"):
            validate_daily_tmax_forecast(df)

    def test_non_utc_issue_time_raises(self, make_daily_tmax_forecast) -> None:
        """Non-UTC issue_time_utc should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        # Set to Eastern timezone
        df["issue_time_utc"] = pd.Timestamp.now(tz="America/New_York")
        with pytest.raises(ValueError, match="(Wrong timezone|UTC)"):
            validate_daily_tmax_forecast(df)

    def test_naive_issue_time_raises(self, make_daily_tmax_forecast) -> None:
        """Timezone-naive issue_time_utc should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        # Set to naive datetime
        df["issue_time_utc"] = pd.Timestamp.now()  # No timezone
        with pytest.raises(ValueError, match="Timezone required"):
            validate_daily_tmax_forecast(df)

    def test_non_utc_ingested_at_raises(self, make_daily_tmax_forecast) -> None:
        """Non-UTC ingested_at_utc should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        # Set to Eastern timezone
        df["ingested_at_utc"] = pd.Timestamp.now(tz="America/New_York")
        with pytest.raises(ValueError, match="(Wrong timezone|UTC)"):
            validate_daily_tmax_forecast(df)

    def test_naive_ingested_at_raises(self, make_daily_tmax_forecast) -> None:
        """Timezone-naive ingested_at_utc should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        # Set to naive datetime
        df["ingested_at_utc"] = pd.Timestamp.now()  # No timezone
        with pytest.raises(ValueError, match="Timezone required"):
            validate_daily_tmax_forecast(df)

    def test_target_date_with_time_component_raises(self, make_daily_tmax_forecast) -> None:
        """target_date_local with non-midnight time should raise ValueError."""
        df = make_daily_tmax_forecast(n_rows=5)
        # Add a time component
        df.loc[2, "target_date_local"] = pd.Timestamp("2024-07-03 14:30:00")
        with pytest.raises(ValueError, match="Date has time component"):
            validate_daily_tmax_forecast(df)
