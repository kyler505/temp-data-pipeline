"""Reporting modules for daily maximum temperature evaluation.

Provides:
- load_run_summary: Load a run's metadata and metrics
- compare_runs: Compare multiple runs side-by-side
- list_all_runs: List all runs with summary metrics
- print_run_comparison: Formatted comparison output
"""

from tempdata.report.report_daily_tmax import (
    compare_runs,
    list_all_runs,
    load_run_summary,
    print_run_comparison,
)

__all__ = [
    "compare_runs",
    "list_all_runs",
    "load_run_summary",
    "print_run_comparison",
]
