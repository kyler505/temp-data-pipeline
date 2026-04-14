#!/bin/bash
# scripts/hprc/01_fetch_data.sh
# Run this on a Login Node or Data Transfer Node.
# Usage: ./scripts/hprc/01_fetch_data.sh <STATION_ID> <TIMEZONE> [START_DATE] [END_DATE]

STATION=${1:-"KLGA"}
TIMEZONE=${2:-"America/New_York"}
START=${3:-"2020-01-01"}
END=${4:-"2024-01-01"}

# Ensure environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Error: Virtual environment not active."
    echo "Run: source ~/envs/temp-pipeline/bin/activate"
    exit 1
fi

# Ensure storage location is set
if [[ -z "$TEMP_DATA_ROOT" ]]; then
    echo "Error: TEMP_DATA_ROOT not set."
    echo "Export it first, e.g.: export TEMP_DATA_ROOT=/scratch/user/$USER/temp-data"
    exit 1
fi

echo "Fetching/building cached datasets for $STATION from $START to $END..."
echo "Timezone: $TIMEZONE"
echo "Data root: $TEMP_DATA_ROOT"

# Recommended: use the orchestrator. This will fetch NOAA hourly truth, clean it,
# build daily_tmax, and *optionally* fetch Open-Meteo historical forecasts (if available)
# and build the forecast-dependent training feature table.
#
# If you want strict forecast coverage enforcement for a given date range, add:
#   --require-forecast

tempdata data \
  --station "$STATION" \
  --start "$START" \
  --end "$END" \
  --timezone "$TIMEZONE" \
  --data-dir "$TEMP_DATA_ROOT/data"

echo "Done."
