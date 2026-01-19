"""Schema definitions for the temp data pipeline.

This package defines the contract layer - what "valid data" looks like.
Nothing here should do work, only define structure.

Schemas:
- qc_flags: Quality control flag bitmasks
- hourly_obs: Canonical hourly observation structure
- daily_tmax: Daily maximum temperature structure
- validate: Validation helpers and validators
"""

from tempdata.schemas.daily_tmax import (
    DAILY_TMAX_FIELDS,
    REQUIRED_COLUMNS as DAILY_TMAX_REQUIRED_COLUMNS,
    DailyTmax,
    validate_daily_tmax,
)
from tempdata.schemas.hourly_obs import (
    RAW_HOURLY_FIELDS,
    REQUIRED_COLUMNS as HOURLY_OBS_REQUIRED_COLUMNS,
    HourlyObs,
    ensure_hourly_schema_columns,
    validate_hourly_obs,
)
from tempdata.schemas.qc_flags import (
    QC_DUPLICATE_TS,
    QC_INCOMPLETE_DAY,
    QC_LOW_COVERAGE,
    QC_MISSING_VALUE,
    QC_OK,
    QC_OUT_OF_RANGE,
    QC_SPIKE_DETECTED,
)
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

__all__ = [
    # QC Flags
    "QC_OK",
    "QC_MISSING_VALUE",
    "QC_OUT_OF_RANGE",
    "QC_SPIKE_DETECTED",
    "QC_DUPLICATE_TS",
    "QC_LOW_COVERAGE",
    "QC_INCOMPLETE_DAY",
    # Hourly Observations
    "HourlyObs",
    "RAW_HOURLY_FIELDS",
    "HOURLY_OBS_REQUIRED_COLUMNS",
    "ensure_hourly_schema_columns",
    "validate_hourly_obs",
    # Daily Tmax
    "DailyTmax",
    "DAILY_TMAX_FIELDS",
    "DAILY_TMAX_REQUIRED_COLUMNS",
    "validate_daily_tmax",
    # Validation helpers
    "require_columns",
    "require_dtypes",
    "require_no_nulls",
    "require_unique",
    "require_timezone_utc",
    "require_range",
    "require_int_range",
    "require_nonnegative_int",
    "require_close",
    "require_date_no_time",
]
