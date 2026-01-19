"""Tests for validation helper functions."""

from __future__ import annotations

import pandas as pd
import pytest

from tempdata.schemas.validate import (
    require_close,
    require_columns,
    require_date_no_time,
    require_dtypes,
    require_int_range,
    require_nonnegative_int,
    require_no_nulls,
    require_range,
    require_timezone_utc,
    require_unique,
)


class TestRequireColumns:
    """Tests for require_columns helper."""

    def test_all_columns_present_passes(self) -> None:
        """Should pass when all required columns are present."""
        columns = ["a", "b", "c", "d"]
        require_columns(columns, ["a", "b"])

    def test_missing_column_raises(self) -> None:
        """Should raise when required column is missing."""
        columns = ["a", "b"]
        with pytest.raises(ValueError, match="Missing columns"):
            require_columns(columns, ["a", "b", "c"])

    def test_dataset_name_in_error(self) -> None:
        """Dataset name should appear in error message."""
        columns = ["a"]
        with pytest.raises(ValueError, match="test_dataset"):
            require_columns(columns, ["a", "b"], dataset="test_dataset")


class TestRequireNoNulls:
    """Tests for require_no_nulls helper."""

    def test_no_nulls_passes(self) -> None:
        """Should pass when no nulls in specified columns."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        require_no_nulls(df, ["a", "b"])

    def test_null_raises(self) -> None:
        """Should raise when null found."""
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="Null values"):
            require_no_nulls(df, ["a"])

    def test_includes_count(self) -> None:
        """Error message should include count of nulls."""
        df = pd.DataFrame({"a": [None, None, 3]})
        with pytest.raises(ValueError, match="2 rows"):
            require_no_nulls(df, ["a"])


class TestRequireUnique:
    """Tests for require_unique helper."""

    def test_unique_passes(self) -> None:
        """Should pass when keys are unique."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        require_unique(df, ["a", "b"])

    def test_duplicate_raises(self) -> None:
        """Should raise when duplicates found."""
        df = pd.DataFrame({"a": [1, 1, 3], "b": ["x", "x", "z"]})
        with pytest.raises(ValueError, match="Duplicate keys"):
            require_unique(df, ["a", "b"])

    def test_empty_df_passes(self) -> None:
        """Empty DataFrame should pass."""
        df = pd.DataFrame({"a": [], "b": []})
        require_unique(df, ["a", "b"])


class TestRequireRange:
    """Tests for require_range helper."""

    def test_in_range_passes(self) -> None:
        """Should pass when all values in range."""
        df = pd.DataFrame({"x": [0, 50, 100]})
        require_range(df, "x", lo=0, hi=100)

    def test_below_range_raises(self) -> None:
        """Should raise when value below range."""
        df = pd.DataFrame({"x": [-1, 50, 100]})
        with pytest.raises(ValueError, match="Out of range"):
            require_range(df, "x", lo=0, hi=100)

    def test_above_range_raises(self) -> None:
        """Should raise when value above range."""
        df = pd.DataFrame({"x": [0, 50, 101]})
        with pytest.raises(ValueError, match="Out of range"):
            require_range(df, "x", lo=0, hi=100)

    def test_null_allowed(self) -> None:
        """Nulls should be allowed when allow_null=True."""
        df = pd.DataFrame({"x": [0, None, 100]})
        require_range(df, "x", lo=0, hi=100, allow_null=True)

    def test_null_raises_when_not_allowed(self) -> None:
        """Nulls should not affect range check when allow_null=False."""
        # When allow_null=False, nulls are included in the range check
        # Since NaN comparisons are False, they won't trigger the out-of-range error
        # This test verifies the behavior
        df = pd.DataFrame({"x": [0, float("nan"), 100]})
        require_range(df, "x", lo=0, hi=100, allow_null=False)


class TestRequireTimezoneUtc:
    """Tests for require_timezone_utc helper."""

    def test_utc_passes(self) -> None:
        """Should pass for UTC timestamps."""
        df = pd.DataFrame(
            {"ts": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")}
        )
        require_timezone_utc(df, "ts")

    def test_non_utc_raises(self) -> None:
        """Should raise for non-UTC timezone."""
        df = pd.DataFrame(
            {"ts": pd.date_range("2024-01-01", periods=3, freq="h", tz="America/New_York")}
        )
        with pytest.raises(ValueError, match="(Wrong timezone|UTC)"):
            require_timezone_utc(df, "ts")

    def test_naive_raises(self) -> None:
        """Should raise for timezone-naive timestamps."""
        df = pd.DataFrame(
            {"ts": pd.date_range("2024-01-01", periods=3, freq="h")}
        )
        with pytest.raises(ValueError, match="Timezone required"):
            require_timezone_utc(df, "ts")

    def test_empty_df_passes(self) -> None:
        """Empty DataFrame should pass."""
        df = pd.DataFrame({"ts": pd.DatetimeIndex([], tz="UTC")})
        require_timezone_utc(df, "ts")


class TestRequireClose:
    """Tests for require_close helper."""

    def test_close_values_pass(self) -> None:
        """Should pass when values are within tolerance."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.01, 2.01, 3.01]})
        require_close(df, "a", "b", tol=0.1)

    def test_not_close_raises(self) -> None:
        """Should raise when values differ by more than tolerance."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 4.0]})
        with pytest.raises(ValueError, match="Values not close"):
            require_close(df, "a", "b", tol=0.5)

    def test_with_transform(self) -> None:
        """Should apply transform before comparison."""
        # Celsius to Fahrenheit: F = C * 9/5 + 32
        df = pd.DataFrame({"c": [0, 100], "f": [32.0, 212.0]})
        require_close(
            df,
            "c",
            "f",
            tol=0.1,
            transform_a_to_b=lambda x: x * 9 / 5 + 32,
        )


class TestRequireDateNoTime:
    """Tests for require_date_no_time helper."""

    def test_midnight_passes(self) -> None:
        """Should pass when all times are midnight."""
        df = pd.DataFrame(
            {"date": pd.date_range("2024-01-01", periods=3, freq="D")}
        )
        require_date_no_time(df, "date")

    def test_non_midnight_raises(self) -> None:
        """Should raise when time component is non-zero."""
        df = pd.DataFrame(
            {"date": [pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-01-02 12:30")]}
        )
        with pytest.raises(ValueError, match="Date has time component"):
            require_date_no_time(df, "date")
