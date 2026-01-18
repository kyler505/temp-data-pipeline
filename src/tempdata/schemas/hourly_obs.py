"""Schema definitions for hourly observations."""

from __future__ import annotations

from typing import Iterable

RAW_HOURLY_FIELDS = [
    "ts_utc",
    "station_id",
    "lat",
    "lon",
    "temp_c",
    "source",
    "qc_flags",
]


def ensure_hourly_schema_columns(columns: Iterable[str]) -> list[str]:
    """Return columns ordered to the raw hourly schema."""
    col_set = set(columns)
    missing = [col for col in RAW_HOURLY_FIELDS if col not in col_set]
    if missing:
        raise ValueError(f"Missing raw hourly fields: {missing}")
    return list(RAW_HOURLY_FIELDS)
