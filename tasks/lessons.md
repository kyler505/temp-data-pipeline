# Lessons

- Smoke-test the real parquet-backed eval path after wiring changes; unit tests alone did not catch the `datetime64[ns]` vs `date` comparison bug in `target_date_local` filtering.
