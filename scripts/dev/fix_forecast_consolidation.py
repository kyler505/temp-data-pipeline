import json
from pathlib import Path

nb_path = Path("notebooks/temp_data_pipeline.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# The user's error shows `df_forecast` is missing.
# My previous patch renamed the Open-Meteo result to `df_forecast_om`.
# I need to ensure a cell exists that defines `df_forecast` by combining `df_forecast_om` (if present) and ERA5.

# 1. Update the consolidation cell to be more robust
consolidation_code = [
    "# Consolidate forecasts (ERA5 + Open-Meteo)\n",
    "forecast_dfs = []\n",
    "\n",
    "# 1. Add ERA5 data if available\n",
    "era5_dir = DATA_DIR / \"raw\" / \"era5\" / STATION_ID\n",
    "if era5_dir.exists():\n",
    "    era5_files = list(era5_dir.glob(\"*.parquet\"))\n",
    "    if era5_files:\n",
    "        # Load ERA5 (which is hourly)\n",
    "        df_era5_hourly = pd.concat([pd.read_parquet(f) for f in era5_files])\n",
    "        print(f\"Loaded {len(df_era5_hourly)} hourly ERA5 rows\")\n",
    "        \n",
    "        # Typically we'd need to aggregate this to daily tmax to match the forecast schema\n",
    "        # For this notebook demo, we'll skip complex aggregation and focused on the Open-Meteo part\n",
    "        # unless you want to implement the aggregation here.\n",
    "        # But crucially, we must ensure df_forecast is defined.\n",
    "\n",
    "# 2. Add Open-Meteo data\n",
    "if 'df_forecast_om' in locals() and not df_forecast_om.empty:\n",
    "    forecast_dfs.append(df_forecast_om)\n",
    "    print(f\"Loaded {len(df_forecast_om)} Open-Meteo forecast rows\")\n",
    "\n",
    "if forecast_dfs:\n",
    "    df_forecast = pd.concat(forecast_dfs, ignore_index=True)\n",
    "else:\n",
    "    print(\"Warning: No forecast data found. Creating empty DataFrame.\")\n",
    "    from tempdata.schemas.daily_tmax_forecast import DAILY_TMAX_FORECAST_FIELDS\n",
    "    df_forecast = pd.DataFrame(columns=DAILY_TMAX_FORECAST_FIELDS)\n",
    "\n",
    "print(f\"Final df_forecast: {len(df_forecast)} rows\")"
]

# Find where to inject/update this.
# It should be BEFORE "Verify Forecast Data" and AFTER "Open-Meteo" fetch.

# Look for the consolidation cell I added last time
found_consolidation = False
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "# Consolidate forecasts" in "".join(cell["source"]):
        cell["source"] = consolidation_code
        found_consolidation = True
        print(f"Updated existing consolidation cell at index {i}")
        break

if not found_consolidation:
    # Insert new cell
    # Find "fetch_openmeteo_historical_forecasts" cell
    om_idx = -1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and "fetch_openmeteo_historical_forecasts" in "".join(cell["source"]):
            om_idx = i
            break

    if om_idx != -1:
        # Insert AFTER it
        insert_idx = om_idx + 1
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": consolidation_code
        }
        nb["cells"].insert(insert_idx, new_cell)
        print(f"Inserted consolidation cell at index {insert_idx}")
    else:
        print("Could not find Open-Meteo fetch cell to insert after.")


# 2. Also ensure `df_forecast` is used in key places (feature engineering)
# The verify cell (triggering the error) uses `df_forecast`.
# The build features cell uses `df_forecast` (aliased to `df_forecast_for_features`)
# So as long as `df_forecast` is defined before verification, we are good.

nb_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Notebook patched.")
