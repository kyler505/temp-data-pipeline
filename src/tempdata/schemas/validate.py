"""Validation helpers for schema enforcement.

These helpers ensure DataFrames conform to expected schemas.
All helpers raise ValueError with actionable messages including:
- Dataset name (if provided)
- Offending columns
- Count of failing rows
- Sample of failing row indices (first 5)
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def _format_error(
    dataset: str | None,
    rule: str,
    detail: str,
    failing_indices: list[Any] | None = None,
    count: int | None = None,
) -> str:
    """Format a validation error message consistently."""
    parts = []
    if dataset:
        parts.append(f"[{dataset}]")
    parts.append(rule)
    parts.append(f": {detail}")
    if count is not None:
        parts.append(f" ({count} rows)")
    if failing_indices:
        sample = failing_indices[:5]
        parts.append(f" | sample indices: {sample}")
    return "".join(parts)


def require_columns(
    df_columns: Iterable[str],
    required: Iterable[str],
    dataset: str | None = None,
) -> None:
    """Raise ValueError if required columns are missing.

    Args:
        df_columns: Column names from a DataFrame (e.g., df.columns)
        required: Required column names
        dataset: Optional dataset name for error messages

    Raises:
        ValueError: If any required columns are missing
    """
    df_set = set(df_columns)
    required_set = set(required)
    missing = required_set - df_set
    if missing:
        raise ValueError(
            _format_error(dataset, "Missing columns", f"{sorted(missing)}")
        )


def require_dtypes(
    df: pd.DataFrame,
    expected: dict[str, type | str | list[type | str]],
    dataset: str | None = None,
) -> None:
    """Raise ValueError if columns have unexpected dtypes.

    This is a soft check - it allows compatible types (e.g., int64 for int).

    Args:
        df: DataFrame to check
        expected: Dict mapping column names to expected dtype(s)
        dataset: Optional dataset name for error messages

    Raises:
        ValueError: If any columns have incompatible dtypes
    """
    mismatches = []
    for col, exp_types in expected.items():
        if col not in df.columns:
            continue  # Let require_columns handle missing columns

        actual = df[col].dtype
        if not isinstance(exp_types, list):
            exp_types = [exp_types]

        matched = False
        for exp in exp_types:
            if isinstance(exp, str):
                # String dtype check (e.g., "datetime64[ns, UTC]")
                if exp in str(actual):
                    matched = True
                    break
            else:
                # Type check
                if pd.api.types.is_dtype_equal(actual, exp):
                    matched = True
                    break
                # Also check if actual is a subtype (e.g., int64 for int)
                try:
                    if pd.api.types.is_integer_dtype(actual) and exp in (int, "int"):
                        matched = True
                        break
                    if pd.api.types.is_float_dtype(actual) and exp in (float, "float"):
                        matched = True
                        break
                    if pd.api.types.is_string_dtype(actual) and exp in (str, "str"):
                        matched = True
                        break
                except (TypeError, AttributeError):
                    pass

        if not matched:
            mismatches.append(f"{col}: expected {exp_types}, got {actual}")

    if mismatches:
        raise ValueError(
            _format_error(dataset, "Dtype mismatch", "; ".join(mismatches))
        )


def require_no_nulls(
    df: pd.DataFrame,
    cols: Iterable[str],
    dataset: str | None = None,
) -> None:
    """Raise ValueError if specified columns contain null values.

    Args:
        df: DataFrame to check
        cols: Column names that must not have nulls
        dataset: Optional dataset name for error messages

    Raises:
        ValueError: If any specified columns have null values
    """
    for col in cols:
        if col not in df.columns:
            continue  # Let require_columns handle missing columns

        null_mask = df[col].isna()
        null_count = null_mask.sum()
        if null_count > 0:
            failing_indices = df.index[null_mask].tolist()
            raise ValueError(
                _format_error(
                    dataset,
                    "Null values",
                    f"column '{col}' has nulls",
                    failing_indices,
                    null_count,
                )
            )


def require_unique(
    df: pd.DataFrame,
    key_cols: list[str],
    dataset: str | None = None,
) -> None:
    """Raise ValueError if key columns have duplicate combinations.

    Args:
        df: DataFrame to check
        key_cols: Column names that form a unique key
        dataset: Optional dataset name for error messages

    Raises:
        ValueError: If duplicate key combinations exist
    """
    if df.empty:
        return

    for col in key_cols:
        if col not in df.columns:
            return  # Let require_columns handle missing columns

    dup_mask = df.duplicated(subset=key_cols, keep=False)
    dup_count = dup_mask.sum()
    if dup_count > 0:
        failing_indices = df.index[dup_mask].tolist()
        raise ValueError(
            _format_error(
                dataset,
                "Duplicate keys",
                f"columns {key_cols} have duplicates",
                failing_indices,
                dup_count,
            )
        )


def require_timezone_utc(
    df: pd.DataFrame,
    ts_col: str,
    dataset: str | None = None,
) -> None:
    """Raise ValueError if timestamp column is not tz-aware UTC.

    Args:
        df: DataFrame to check
        ts_col: Name of the timestamp column
        dataset: Optional dataset name for error messages

    Raises:
        ValueError: If column is not tz-aware or not UTC
    """
    if ts_col not in df.columns:
        return  # Let require_columns handle missing columns

    if df.empty:
        return

    dtype = df[ts_col].dtype
    # Check if it's a datetime dtype with timezone
    if not hasattr(dtype, "tz") or dtype.tz is None:
        raise ValueError(
            _format_error(
                dataset,
                "Timezone required",
                f"column '{ts_col}' must be tz-aware UTC, got {dtype}",
            )
        )

    # Check if timezone is UTC
    tz_str = str(dtype.tz).upper()
    if tz_str not in ("UTC", "TIMEZONE.UTC", "PYTZ.UTC", "ZONEINFO.ZONEINFO('UTC')"):
        raise ValueError(
            _format_error(
                dataset,
                "Wrong timezone",
                f"column '{ts_col}' must be UTC, got {dtype.tz}",
            )
        )


def require_range(
    df: pd.DataFrame,
    col: str,
    lo: float,
    hi: float,
    allow_null: bool = False,
    dataset: str | None = None,
) -> None:
    """Raise ValueError if values are outside the specified range.

    Args:
        df: DataFrame to check
        col: Column name to check
        lo: Minimum allowed value (inclusive)
        hi: Maximum allowed value (inclusive)
        allow_null: If True, null values are allowed
        dataset: Optional dataset name for error messages

    Raises:
        ValueError: If values are outside range
    """
    if col not in df.columns:
        return  # Let require_columns handle missing columns

    if df.empty:
        return

    series = df[col]
    if allow_null:
        series = series.dropna()

    out_of_range = (series < lo) | (series > hi)
    bad_count = out_of_range.sum()
    if bad_count > 0:
        failing_indices = series.index[out_of_range].tolist()
        raise ValueError(
            _format_error(
                dataset,
                "Out of range",
                f"column '{col}' must be in [{lo}, {hi}]",
                failing_indices,
                bad_count,
            )
        )


def require_int_range(
    df: pd.DataFrame,
    col: str,
    lo: int,
    hi: int,
    allow_null: bool = False,
    dataset: str | None = None,
) -> None:
    """Raise ValueError if integer values are outside the specified range.

    Args:
        df: DataFrame to check
        col: Column name to check
        lo: Minimum allowed value (inclusive)
        hi: Maximum allowed value (inclusive)
        allow_null: If True, null values are allowed
        dataset: Optional dataset name for error messages

    Raises:
        ValueError: If values are outside range or not integer-like
    """
    if col not in df.columns:
        return  # Let require_columns handle missing columns

    if df.empty:
        return

    series = df[col]
    if allow_null:
        non_null = series.dropna()
    else:
        non_null = series

    # Check for integer-like values
    if not non_null.empty:
        # Check range
        out_of_range = (non_null < lo) | (non_null > hi)
        bad_count = out_of_range.sum()
        if bad_count > 0:
            failing_indices = non_null.index[out_of_range].tolist()
            raise ValueError(
                _format_error(
                    dataset,
                    "Out of range",
                    f"column '{col}' must be in [{lo}, {hi}]",
                    failing_indices,
                    bad_count,
                )
            )


def require_nonnegative_int(
    df: pd.DataFrame,
    col: str,
    dataset: str | None = None,
) -> None:
    """Raise ValueError if values are not non-negative integers.

    Args:
        df: DataFrame to check
        col: Column name to check
        dataset: Optional dataset name for error messages

    Raises:
        ValueError: If values are negative
    """
    if col not in df.columns:
        return  # Let require_columns handle missing columns

    if df.empty:
        return

    series = df[col].dropna()
    negative = series < 0
    bad_count = negative.sum()
    if bad_count > 0:
        failing_indices = series.index[negative].tolist()
        raise ValueError(
            _format_error(
                dataset,
                "Negative values",
                f"column '{col}' must be >= 0",
                failing_indices,
                bad_count,
            )
        )


def require_close(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    tol: float,
    transform_a_to_b: callable | None = None,
    dataset: str | None = None,
) -> None:
    """Raise ValueError if two columns are not close within tolerance.

    Useful for checking Celsius/Fahrenheit consistency.

    Args:
        df: DataFrame to check
        col_a: First column name
        col_b: Second column name
        tol: Tolerance for comparison
        transform_a_to_b: Optional function to transform col_a values
            before comparison (e.g., C to F conversion)
        dataset: Optional dataset name for error messages

    Raises:
        ValueError: If values are not close within tolerance
    """
    if col_a not in df.columns or col_b not in df.columns:
        return  # Let require_columns handle missing columns

    if df.empty:
        return

    a_vals = df[col_a]
    b_vals = df[col_b]

    if transform_a_to_b is not None:
        a_vals = transform_a_to_b(a_vals)

    # Only compare where both are non-null
    mask = a_vals.notna() & b_vals.notna()
    diff = (a_vals[mask] - b_vals[mask]).abs()
    bad_mask = diff > tol

    bad_count = bad_mask.sum()
    if bad_count > 0:
        failing_indices = diff.index[bad_mask].tolist()
        raise ValueError(
            _format_error(
                dataset,
                "Values not close",
                f"columns '{col_a}' and '{col_b}' differ by more than {tol}",
                failing_indices,
                bad_count,
            )
        )


def require_date_no_time(
    df: pd.DataFrame,
    col: str,
    dataset: str | None = None,
) -> None:
    """Raise ValueError if datetime column has non-midnight times.

    For date columns stored as datetime, ensures they're at 00:00:00.

    Args:
        df: DataFrame to check
        col: Column name to check
        dataset: Optional dataset name for error messages

    Raises:
        ValueError: If any values have non-zero time components
    """
    if col not in df.columns:
        return  # Let require_columns handle missing columns

    if df.empty:
        return

    series = df[col].dropna()

    # If it's already a date type, no check needed
    if pd.api.types.is_object_dtype(series):
        return

    # Check if datetime and verify midnight
    if pd.api.types.is_datetime64_any_dtype(series):
        # Remove timezone for comparison
        if hasattr(series.dt, "tz") and series.dt.tz is not None:
            series_local = series.dt.tz_localize(None)
        else:
            series_local = series

        has_time = (
            (series_local.dt.hour != 0)
            | (series_local.dt.minute != 0)
            | (series_local.dt.second != 0)
        )
        bad_count = has_time.sum()
        if bad_count > 0:
            failing_indices = series.index[has_time].tolist()
            raise ValueError(
                _format_error(
                    dataset,
                    "Date has time component",
                    f"column '{col}' should be midnight (00:00:00)",
                    failing_indices,
                    bad_count,
                )
            )
