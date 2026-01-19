
import json
import pathlib

def fix_notebook():
    nb_path = pathlib.Path("notebooks/backtest_analysis.ipynb")
    if not nb_path.exists():
        print(f"Notebook not found: {nb_path}")
        return

    with open(nb_path, "r") as f:
        nb = json.load(f)

    # Find the cell that loads artifacts (Section 5)
    target_cell_idx = -1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "# Load DataFrames" in source and "pd.read_parquet" in source:
                target_cell_idx = i
                break

    if target_cell_idx == -1:
        print("Could not find data loading cell")
        return

    print(f"Found target cell at index {target_cell_idx}")
    cell = nb["cells"][target_cell_idx]

    # Check if fix is already applied
    source_str = "".join(cell["source"])
    if "# FIXES FOR ANALYSIS - V2" in source_str:
        print("Fix V2 already applied")
        return

    # Remove previous fix if it exists (simple check or just append new one? Appending is safer to avoid breaking if logic changed)
    # But since we want to encompass all fixes, let's just append a comprehensive block and maybe wrap it common

    # Prepare the fix code
    fix_code = [
        "\n",
        "# ==========================================\n",
        "# FIXES FOR ANALYSIS COMPATIBILITY - V2\n",
        "# ==========================================\n",
        "if daily_df is not None:\n",
        "    # 1. Ensure 'date' column exists\n",
        "    if 'target_date_local' in daily_df.columns and 'date' not in daily_df.columns:\n",
        "        daily_df['date'] = pd.to_datetime(daily_df['target_date_local'])\n",
        "    \n",
        "    # 2. Calculate bankroll_end if missing\n",
        "    if 'bankroll_end' not in daily_df.columns and 'total_pnl' in daily_df.columns:\n",
        "        initial_bankroll = config.get('initial_bankroll', 10000.0)\n",
        "        daily_df = daily_df.sort_values('date')\n",
        "        daily_df['cumulative_pnl'] = daily_df['total_pnl'].cumsum()\n",
        "        daily_df['bankroll_end'] = initial_bankroll + daily_df['cumulative_pnl']\n",
        "    \n",
        "    # 3. Calculate drawdown if missing\n",
        "    if 'drawdown' not in daily_df.columns and 'bankroll_end' in daily_df.columns:\n",
        "        daily_df['peak'] = daily_df['bankroll_end'].cummax()\n",
        "        daily_df['drawdown'] = daily_df['peak'] - daily_df['bankroll_end']\n",
        "\n",
        "if predictions_df is not None:\n",
        "    # 4. Alias sigma\n",
        "    if 'sigma_f' in predictions_df.columns and 'sigma' not in predictions_df.columns:\n",
        "        predictions_df['sigma'] = predictions_df['sigma_f']\n",
        "    \n",
        "    # 5. Calculate residual\n",
        "    if 'residual' not in predictions_df.columns and 'tmax_actual_f' in predictions_df.columns:\n",
        "        predictions_df['mu_f'] = pd.to_numeric(predictions_df['mu_f'], errors='coerce')\n",
        "        predictions_df['tmax_actual_f'] = pd.to_numeric(predictions_df['tmax_actual_f'], errors='coerce')\n",
        "        predictions_df['residual'] = predictions_df['mu_f'] - predictions_df['tmax_actual_f']\n",
        "\n",
        "    # 6. Calculate MAE and RMSE columns (required for monthly aggregation)\n",
        "    if 'residual' in predictions_df.columns:\n",
        "        if 'mae' not in predictions_df.columns:\n",
        "            predictions_df['mae'] = predictions_df['residual'].abs()\n",
        "        if 'rmse' not in predictions_df.columns:\n",
        "            predictions_df['rmse'] = predictions_df['residual'] ** 2\n",
        "    \n",
        "    print(f\"Applied V2 compatibility fixes: added date, bankroll_end, drawdown, sigma, residual, mae, rmse columns\")\n"
    ]

    # Append fix code to the cell
    cell["source"].extend(fix_code)

    # Write back
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)

    print("Notebook patched successfully with V2 fixes")

if __name__ == "__main__":
    fix_notebook()
