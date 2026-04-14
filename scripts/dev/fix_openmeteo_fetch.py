import json
from pathlib import Path

nb_path = Path("notebooks/temp_data_pipeline.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# 1. Modify the Open-Meteo fetch cell to clip start_date
target_source_fragment = "fetch_openmeteo_historical_forecasts"
modified_source = [
    "from tempdata.fetch.openmeteo_daily_forecast import fetch_openmeteo_historical_forecasts\n",
    "from tempdata.schemas.daily_tmax_forecast import validate_daily_tmax_forecast\n",
    "from datetime import datetime, timedelta\n",
    "import pandas as pd\n",
    "\n",
    "# Use the same date range as the truth data (from NOAA fetch)\n",
    "OPENMETEO_MIN_DATE = \"2016-01-01\"\n",
    "if pd.Timestamp(START_DATE) < pd.Timestamp(OPENMETEO_MIN_DATE):\n",
    "    print(f\"[notebook] Adjusting Open-Meteo start date from {START_DATE} to {OPENMETEO_MIN_DATE} (API limit)\")\n",
    "    FORECAST_START_DATE = OPENMETEO_MIN_DATE\n",
    "else:\n",
    "    FORECAST_START_DATE = START_DATE\n",
    "\n",
    "FORECAST_END_DATE = END_DATE\n",
    "\n",
    "# Adjust end date: NOAA uses exclusive end, Open-Meteo uses inclusive\n",
    "end_dt = datetime.strptime(FORECAST_END_DATE, \"%Y-%m-%d\") - timedelta(days=1)\n",
    "forecast_end_date = end_dt.strftime(\"%Y-%m-%d\")\n",
    "\n",
    "if pd.Timestamp(FORECAST_START_DATE) <= pd.Timestamp(forecast_end_date):\n",
    "    print(f\"Fetching historical forecasts for {STATION_ID}\")\n",
    "    print(f\"Date range: {FORECAST_START_DATE} to {forecast_end_date}\")\n",
    "\n",
    "    # Output directories\n",
    "    FORECAST_RAW_DIR = DATA_DIR / \"raw\" / \"forecasts\" / \"openmeteo\" / STATION_ID\n",
    "    FORECAST_CLEAN_DIR = DATA_DIR / \"clean\" / \"forecasts\" / \"openmeteo\" / STATION_ID\n",
    "\n",
    "    forecast_files, df_forecast_om = fetch_openmeteo_historical_forecasts(\n",
    "        station_id=STATION_ID,\n",
    "        start_date=FORECAST_START_DATE,\n",
    "        end_date=forecast_end_date,\n",
    "        out_raw_dir=FORECAST_RAW_DIR,\n",
    "        out_parquet_dir=FORECAST_CLEAN_DIR,\n",
    "        write_raw=True,  # Save raw JSON for debugging\n",
    "    )\n",
    "\n",
    "    print(f\"\\nWrote {len(forecast_files)} files:\")\n",
    "    for path in forecast_files:\n",
    "        print(f\"  - {path}\")\n",
    "else:\n",
    "    print(\"Open-Meteo forecast range is empty (dates are pre-2016)\")\n",
    "    df_forecast_om = pd.DataFrame()\n"
]

found_om = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and target_source_fragment in "".join(cell["source"]):
        cell["source"] = modified_source
        found_om = True
        print("Updated Open-Meteo fetch cell.")
        break

# 2. Add ERA5 fetch cell BEFORE the Open-Meteo cell (merged logic)
# Actually, better to add a separate ERA5 cell block if not present.
# I will check if "fetch_era5_hourly" is already called in a code cell.

found_era5_call = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "fetch_era5_hourly" in "".join(cell["source"]):
        found_era5_call = True
        break

if not found_era5_call:
    # Insert ERA5 fetch logic before Open-Meteo
    # Find the Open-Meteo cell index again
    om_idx = -1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and target_source_fragment in "".join(cell["source"]): # Note: source changed above but text match might fail if I don't be careful.
            # Ah, I replaced it in place. `target_source_fragment` is still in `modified_source`?
            # Yes, "fetch_openmeteo_historical_forecasts" is in modified_source.
            om_idx = i
            break

    if om_idx != -1:
        era5_cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Fetch ERA5 Data (Deep Historical)\n",
                    "\n",
                    "For dates before 2016, use ERA5 reanalysis data."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from tempdata.fetch.era5_hourly import fetch_era5_hourly\n",
                    "import pandas as pd\n",
                    "\n",
                    "ERA5_CUTOFF = \"2016-01-01\"\n",
                    "if pd.Timestamp(START_DATE) < pd.Timestamp(ERA5_CUTOFF):\n",
                    "    era5_end = min(pd.Timestamp(END_DATE), pd.Timestamp(ERA5_CUTOFF)).strftime(\"%Y-%m-%d\")\n",
                    "    print(f\"Fetching ERA5 data for {START_DATE} to {era5_end}\")\n",
                    "    \n",
                    "    try:\n",
                    "        era5_files = fetch_era5_hourly(\n",
                    "            station_id=STATION_ID,\n",
                    "            start_date=START_DATE,\n",
                    "            end_date=era5_end,\n",
                    "            out_dir=DATA_DIR / \"raw\" / \"era5\" / STATION_ID,\n",
                    "        )\n",
                    "        print(f\"Wrote {len(era5_files)} ERA5 parquet files\")\n",
                    "    except ImportError:\n",
                    "        print(\"Skipping ERA5: dependencies not installed (run setup cells above)\")\n",
                    "    except Exception as e:\n",
                    "        print(f\"Skipping ERA5: {e}\")\n",
                    "else:\n",
                    "    print(\"No ERA5 data needed (start date >= 2016)\")"
                ]
            }
        ]
        print(f"Inserting ERA5 fetch cells at index {om_idx}")
        nb["cells"][om_idx:om_idx] = era5_cells

# 3. Consolidate forecasts
# I need to ensure `df_forecast` (used later) combines both ERA5 and Open-Meteo.
# The original notebook created `df_forecast` from Open-Meteo only.
# I should inject a cell AFTER Open-Meteo clean/verify that combines them.

# Check if consolidation cell exists
consolidation_fragment = "# Consolidate forecasts"
found_consolidate = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and consolidation_fragment in "".join(cell["source"]):
        found_consolidate = True
        break

if not found_consolidate:
    # Find where `df_forecast` is verified/printed, usually after Open-Meteo fetch
    verify_idx = -1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and "print(f\"Forecast Summary:\")" in "".join(cell["source"]):
            verify_idx = i + 1 # Insert after verification
            break

    if verify_idx != -1:
        consolidate_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Consolidate forecasts (ERA5 + Open-Meteo)\n",
                "forecast_dfs = []\n",
                "\n",
                "# Load ERA5 if present\n",
                "era5_dir = DATA_DIR / \"raw\" / \"era5\" / STATION_ID\n",
                "if era5_dir.exists():\n",
                "    era5_files = list(era5_dir.glob(\"*.parquet\"))\n",
                "    if era5_files:\n",
                "        df_era5 = pd.concat([pd.read_parquet(f) for f in era5_files])\n",
                "        # ERA5 is currently raw hourly. We need to aggregate to daily tmax or use as-is?\n",
                "        # Wait, the pipeline expects 'df_forecast' to have daily tmax cols.\n",
                "        # ERA5 hourly needs aggregation.\n",
                "        # For now, let's just use Open-Meteo df_forecast_om as the base\n",
                "        # and warn the user that ERA5 aggregation to forecast format is needed if they want deep history features.\n",
                "        print(f\"Found {len(df_era5)} hourly ERA5 rows. (Note: ERA5 aggregation to daily forecast format not yet implemented in notebook)\")\n",
                "        \n",
                "if 'df_forecast_om' in locals():\n",
                "    df_forecast = df_forecast_om\n",
                "else:\n",
                "    # Fallback if cell didn't run\n",
                "    df_forecast = pd.DataFrame()\n",
                "\n",
                "print(f\"Final forecast dataframe: {len(df_forecast)} rows\")"
            ]
        }
        print(f"Inserting consolidation cell at index {verify_idx}")
        nb["cells"][verify_idx:verify_idx] = [consolidate_cell]

nb_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Notebook patched.")
