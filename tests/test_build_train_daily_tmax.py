"""Tests for train_daily_tmax feature engineering.

Key tests:
- Rolling features use shift(1) for no lookahead
- Seasonal encodings compute correctly
- Join filters low-coverage days
- Schema validation passes
- Duplicate rejection works
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from tempdata.features.build_train_daily_tmax import (
    add_seasonal_features,
    build_train_daily_tmax,
    join_forecast_to_truth,
)
from tempdata.features.rolling_stats import (
    compute_all_rolling_features,
    compute_rolling_bias,
    compute_rolling_rmse,
    compute_sigma_lead,
)
from tempdata.schemas.train_daily_tmax import (
    REQUIRED_COLUMNS,
    validate_train_daily_tmax,
)


class TestSeasonalFeatures:
    """Tests for seasonal and calendar encodings."""

    def test_sin_cos_doy_range(self) -> None:
        """sin_doy and cos_doy should be in [-1, 1]."""
        # Create dates spanning a full year
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        df = pd.DataFrame({"target_date_local": dates})

        result = add_seasonal_features(df)

        assert result["sin_doy"].min() >= -1
        assert result["sin_doy"].max() <= 1
        assert result["cos_doy"].min() >= -1
        assert result["cos_doy"].max() <= 1

    def test_sin_doy_peaks_summer_solstice(self) -> None:
        """sin_doy should peak near summer solstice (day ~172)."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        df = pd.DataFrame({"target_date_local": dates})

        result = add_seasonal_features(df)

        # Find peak of sin_doy
        peak_idx = result["sin_doy"].idxmax()
        peak_doy = result.loc[peak_idx, "target_date_local"].dayofyear

        # Should be around day 91-92 (spring equinox + 90 days ~ summer)
        # sin(2*pi*doy/365) peaks when doy ≈ 365/4 ≈ 91
        assert 85 <= peak_doy <= 100

    def test_cos_doy_peaks_jan_1(self) -> None:
        """cos_doy should peak near Jan 1 (day 1)."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        df = pd.DataFrame({"target_date_local": dates})

        result = add_seasonal_features(df)

        # cos_doy should be highest on day 1
        jan1_cos = result.loc[0, "cos_doy"]
        assert jan1_cos > 0.99  # Very close to 1

    def test_month_extraction(self) -> None:
        """Month should be correctly extracted as 1-12."""
        dates = [
            pd.Timestamp("2024-01-15"),
            pd.Timestamp("2024-06-15"),
            pd.Timestamp("2024-12-15"),
        ]
        df = pd.DataFrame({"target_date_local": dates})

        result = add_seasonal_features(df)

        assert list(result["month"]) == [1, 6, 12]


class TestRollingStats:
    """Tests for rolling bias and error statistics."""

    def _make_residual_df(
        self,
        residuals: list[float],
        station_id: str = "KLGA",
        lead_hours: int = 24,
    ) -> pd.DataFrame:
        """Create a DataFrame with residuals for testing."""
        n = len(residuals)
        return pd.DataFrame(
            {
                "station_id": station_id,
                "lead_hours": lead_hours,
                "issue_time_utc": pd.date_range(
                    "2024-01-01", periods=n, freq="D", tz="UTC"
                ),
                "residual": residuals,
            }
        )

    def test_rolling_bias_excludes_current_row(self) -> None:
        """Rolling bias should use shift(1), excluding current row."""
        # Residuals: 0, 1, 2, 3, 4
        # With shift(1) and window=2:
        # Row 0: NaN (no prior data)
        # Row 1: mean([0]) = 0 (only one prior, min_periods=1)
        # Row 2: mean([0, 1]) = 0.5
        # Row 3: mean([1, 2]) = 1.5
        # Row 4: mean([2, 3]) = 2.5
        df = self._make_residual_df([0.0, 1.0, 2.0, 3.0, 4.0])
        df = df.sort_values("issue_time_utc")

        result = compute_rolling_bias(df, windows=[2])

        assert pd.isna(result.loc[0, "bias_2d"])
        assert result.loc[1, "bias_2d"] == pytest.approx(0.0)
        assert result.loc[2, "bias_2d"] == pytest.approx(0.5)
        assert result.loc[3, "bias_2d"] == pytest.approx(1.5)
        assert result.loc[4, "bias_2d"] == pytest.approx(2.5)

    def test_rolling_bias_no_lookahead(self) -> None:
        """The last row's bias should not include its own residual."""
        df = self._make_residual_df([1.0, 1.0, 1.0, 1.0, 100.0])

        result = compute_rolling_bias(df, windows=[7])

        # The last row has residual=100, but bias_7d should be 1.0
        # (mean of all prior 1.0 values)
        assert result.loc[4, "bias_7d"] == pytest.approx(1.0)

    def test_rolling_rmse_excludes_current_row(self) -> None:
        """Rolling RMSE should use shift(1), excluding current row."""
        # Residuals: 1, 1, 1
        # RMSE with shift(1) and window=2:
        # Row 0: NaN
        # Row 1: sqrt(mean([1^2])) = 1.0
        # Row 2: sqrt(mean([1^2, 1^2])) = 1.0
        df = self._make_residual_df([1.0, 1.0, 1.0])

        result = compute_rolling_rmse(df, windows=[2])

        assert pd.isna(result.loc[0, "rmse_2d"])
        assert result.loc[1, "rmse_2d"] == pytest.approx(1.0)
        assert result.loc[2, "rmse_2d"] == pytest.approx(1.0)

    def test_sigma_lead_uses_expanding_window(self) -> None:
        """sigma_lead should use all prior data, not fixed window."""
        # Residuals with known std
        df = self._make_residual_df([0.0, 2.0, 4.0, 6.0, 8.0])

        result = compute_sigma_lead(df)

        # Row 0: NaN (no prior data)
        # Row 1: NaN (only 1 prior, need min_periods=2)
        # Row 2: std([0, 2]) = 1.414...
        # Row 3: std([0, 2, 4]) = 2.0
        # Row 4: std([0, 2, 4, 6]) = 2.58...
        assert pd.isna(result.loc[0, "sigma_lead"])
        assert pd.isna(result.loc[1, "sigma_lead"])
        assert result.loc[2, "sigma_lead"] == pytest.approx(np.std([0, 2], ddof=1))
        assert result.loc[3, "sigma_lead"] == pytest.approx(np.std([0, 2, 4], ddof=1))

    def test_groupby_station_and_lead(self) -> None:
        """Rolling stats should be computed per (station_id, lead_hours) group."""
        # Two stations with different residual patterns
        df1 = self._make_residual_df([1.0, 1.0, 1.0], station_id="KLGA")
        df2 = self._make_residual_df([10.0, 10.0, 10.0], station_id="KJFK")
        df = pd.concat([df1, df2], ignore_index=True)

        result = compute_rolling_bias(df, windows=[7])

        # KLGA should have bias ~1, KJFK should have bias ~10
        klga_bias = result[result["station_id"] == "KLGA"]["bias_7d"].dropna()
        kjfk_bias = result[result["station_id"] == "KJFK"]["bias_7d"].dropna()

        assert all(klga_bias < 2)
        assert all(kjfk_bias > 8)


class TestJoinForecastToTruth:
    """Tests for joining forecast and truth data."""

    def test_join_matches_on_station_and_date(
        self, make_daily_tmax, make_daily_tmax_forecast
    ) -> None:
        """Join should match on station_id and target_date_local."""
        truth = make_daily_tmax(n_rows=7, station_id="KLGA")
        forecast = make_daily_tmax_forecast(n_rows=7, station_id="KLGA")

        result = join_forecast_to_truth(forecast, truth)

        assert len(result) == 7
        assert "tmax_actual_f" in result.columns
        assert "residual" in result.columns

    def test_join_filters_low_coverage(
        self, make_daily_tmax, make_daily_tmax_forecast
    ) -> None:
        """Join should exclude truth rows with low coverage."""
        truth = make_daily_tmax(n_rows=7, station_id="KLGA")
        forecast = make_daily_tmax_forecast(n_rows=7, station_id="KLGA")

        # Set some days to low coverage
        truth.loc[2, "coverage_hours"] = 10
        truth.loc[4, "coverage_hours"] = 5

        result = join_forecast_to_truth(forecast, truth, min_coverage_hours=18)

        # Should have 5 rows (7 - 2 low coverage)
        assert len(result) == 5

    def test_join_inner_only_matching_dates(
        self, make_daily_tmax, make_daily_tmax_forecast
    ) -> None:
        """Join should only include dates present in both datasets."""
        truth = make_daily_tmax(
            n_rows=5, station_id="KLGA", start_date=datetime(2024, 7, 1)
        )
        forecast = make_daily_tmax_forecast(
            n_rows=5, station_id="KLGA", start_date=datetime(2024, 7, 3)
        )

        result = join_forecast_to_truth(forecast, truth)

        # Only 3 overlapping days: July 3, 4, 5
        assert len(result) == 3

    def test_residual_is_forecast_minus_observed(
        self, make_daily_tmax, make_daily_tmax_forecast
    ) -> None:
        """residual should be (forecast - observed)."""
        truth = make_daily_tmax(n_rows=3, station_id="KLGA", tmax_base=30.0)
        forecast = make_daily_tmax_forecast(n_rows=3, station_id="KLGA", tmax_base=32.0)

        result = join_forecast_to_truth(forecast, truth)

        # Forecast is ~32C = ~89.6F, truth is ~30C = ~86F
        # Residual should be positive (forecast warm bias)
        assert all(result["residual"] > 0)


class TestBuildTrainDailyTmax:
    """Tests for the full feature engineering pipeline."""

    def test_output_has_all_columns(
        self, make_daily_tmax, make_daily_tmax_forecast
    ) -> None:
        """Output should have all required columns."""
        truth = make_daily_tmax(n_rows=30, station_id="KLGA")
        forecast = make_daily_tmax_forecast(n_rows=30, station_id="KLGA")

        result = build_train_daily_tmax(forecast, truth, validate=False)

        for col in REQUIRED_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_validates(
        self, make_daily_tmax, make_daily_tmax_forecast
    ) -> None:
        """Output should pass schema validation."""
        truth = make_daily_tmax(n_rows=30, station_id="KLGA")
        forecast = make_daily_tmax_forecast(n_rows=30, station_id="KLGA")

        result = build_train_daily_tmax(forecast, truth, validate=True)

        # Should not raise
        validate_train_daily_tmax(result)

    def test_empty_input_returns_empty(
        self, make_daily_tmax, make_daily_tmax_forecast
    ) -> None:
        """Empty input should return empty DataFrame with correct columns."""
        truth = make_daily_tmax(n_rows=0)
        forecast = make_daily_tmax_forecast(n_rows=0)

        result = build_train_daily_tmax(forecast, truth)

        assert len(result) == 0
        assert list(result.columns) == REQUIRED_COLUMNS

    def test_drop_warmup_nulls_removes_nan_rows(
        self, make_daily_tmax, make_daily_tmax_forecast
    ) -> None:
        """drop_warmup_nulls=True should remove rows with NaN rolling features."""
        truth = make_daily_tmax(n_rows=30, station_id="KLGA")
        forecast = make_daily_tmax_forecast(n_rows=30, station_id="KLGA")

        result_with_nulls = build_train_daily_tmax(
            forecast, truth, drop_warmup_nulls=False
        )
        result_no_nulls = build_train_daily_tmax(
            forecast, truth, drop_warmup_nulls=True
        )

        # With drop_warmup_nulls, should have fewer or equal rows
        assert len(result_no_nulls) <= len(result_with_nulls)

        # No NaNs in rolling columns
        rolling_cols = ["bias_7d", "bias_14d", "bias_30d", "rmse_14d", "rmse_30d", "sigma_lead"]
        for col in rolling_cols:
            assert result_no_nulls[col].notna().all()

    def test_no_duplicates_in_output(
        self, make_daily_tmax, make_daily_tmax_forecast
    ) -> None:
        """Output should have no duplicate (station_id, issue_time_utc, target_date_local)."""
        truth = make_daily_tmax(n_rows=30, station_id="KLGA")
        forecast = make_daily_tmax_forecast(n_rows=30, station_id="KLGA")

        result = build_train_daily_tmax(forecast, truth)

        key_cols = ["station_id", "issue_time_utc", "target_date_local"]
        duplicates = result.duplicated(subset=key_cols)
        assert not duplicates.any()


class TestSchemaValidation:
    """Tests for train_daily_tmax schema validation."""

    def test_duplicate_raises(
        self, make_daily_tmax, make_daily_tmax_forecast
    ) -> None:
        """Duplicate keys should raise ValueError."""
        truth = make_daily_tmax(n_rows=10, station_id="KLGA")
        forecast = make_daily_tmax_forecast(n_rows=10, station_id="KLGA")

        result = build_train_daily_tmax(forecast, truth, validate=False)

        # Create duplicate
        result = pd.concat([result, result.iloc[[0]]], ignore_index=True)

        with pytest.raises(ValueError, match="Duplicate keys"):
            validate_train_daily_tmax(result)

    def test_out_of_range_temperature_raises(
        self, make_daily_tmax, make_daily_tmax_forecast
    ) -> None:
        """Temperature out of range should raise ValueError."""
        truth = make_daily_tmax(n_rows=10, station_id="KLGA")
        forecast = make_daily_tmax_forecast(n_rows=10, station_id="KLGA")

        result = build_train_daily_tmax(forecast, truth, validate=False)
        result.loc[0, "tmax_pred_f"] = 200  # Out of range

        with pytest.raises(ValueError, match="Out of range"):
            validate_train_daily_tmax(result)

    def test_null_core_column_raises(
        self, make_daily_tmax, make_daily_tmax_forecast
    ) -> None:
        """Null in core column should raise ValueError."""
        truth = make_daily_tmax(n_rows=10, station_id="KLGA")
        forecast = make_daily_tmax_forecast(n_rows=10, station_id="KLGA")

        result = build_train_daily_tmax(forecast, truth, validate=False)
        result.loc[0, "tmax_pred_f"] = None

        with pytest.raises(ValueError, match="Null values"):
            validate_train_daily_tmax(result)

    def test_warmup_nulls_allowed_by_default(
        self, make_daily_tmax, make_daily_tmax_forecast
    ) -> None:
        """NaN in rolling columns should be allowed with allow_warmup_nulls=True."""
        truth = make_daily_tmax(n_rows=10, station_id="KLGA")
        forecast = make_daily_tmax_forecast(n_rows=10, station_id="KLGA")

        result = build_train_daily_tmax(forecast, truth, validate=False)

        # Should not raise (warmup nulls allowed by default)
        validate_train_daily_tmax(result, allow_warmup_nulls=True)
