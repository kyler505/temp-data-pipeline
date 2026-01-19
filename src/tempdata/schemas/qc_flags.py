"""Quality control flag definitions using bitmasks.

This module defines the vocabulary for data quality issues. Multiple issues
can be tracked simultaneously using bitwise OR operations.

Rules:
- Never delete data here
- Only label problems
- Downstream code decides what to exclude
"""

# Base flag: no issues detected
QC_OK = 0

# Hourly-level flags
QC_MISSING_VALUE = 1 << 0  # Temperature value is missing/null
QC_OUT_OF_RANGE = 1 << 1   # Temperature outside reasonable bounds
QC_SPIKE_DETECTED = 1 << 2  # Sudden temperature change detected
QC_DUPLICATE_TS = 1 << 3    # Duplicate timestamp detected

# Daily aggregation flags
QC_LOW_COVERAGE = 1 << 4    # Insufficient hourly observations for day
QC_INCOMPLETE_DAY = 1 << 5  # Day missing expected hours
