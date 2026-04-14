import json
from pathlib import Path

nb_path = Path("notebooks/eval_daily_tmax_analysis.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# 1. Add save_fig helper function
# We'll add this after the valid run detection block (Cell 14 in previous view)
# Ideally, we put it in the "Loading Data" section or a utility cell.
# Let's verify where cell 14 is. It was loading run data.
# We'll insert a wrapper function cell right after it.

save_fig_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "def save_run_plot(fig, filename, run_id, base_dir=Path('../runs')):\n",
  "    \"\"\"Save figure to the run's plot directory.\"\"\"\n",
  "    if not run_id:\n",
  "        return\n",
  "    \n",
  "    plot_dir = base_dir / run_id / \"plots\"\n",
  "    plot_dir.mkdir(exist_ok=True, parents=True)\n",
  "    \n",
  "    out_path = plot_dir / filename\n",
  "    fig.savefig(out_path, dpi=300, bbox_inches='tight')\n",
  "    print(f\"Saved plot: {out_path}\")\n"
 ]
}

# Find index of cell loading run data (contains "Loaded run:")
load_run_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "load_run(run_id" in "".join(cell["source"]):
        load_run_idx = i
        break

if load_run_idx != -1:
    nb["cells"].insert(load_run_idx + 1, save_fig_cell)
    print("Inserted save_fig helper function.")

# 2. Update Plotting Cells
# Cell 16: Residuals
# Cell 17: MAE by Month
# Cell 18: MAE by Season
# Cell 19: Calibration
# Cell 20: Lead Time (Need to check if it exists in file view, assumed yes)

def append_save_call(source, filename):
    # Check if plt.show() exists
    has_show = False
    for i, line in enumerate(source):
        if "plt.show()" in line:
            # Insert save call BEFORE plt.show()
            source.insert(i, f"    save_run_plot(fig, '{filename}', run_id)\n")
            has_show = True
            break
    if not has_show:
         source.append(f"\n    save_run_plot(fig, '{filename}', run_id)\n")
    return source

# Modify plotting cells based on content identification
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src_text = "".join(cell["source"])

        if "Residual Distribution" in src_text and "Q-Q Plot" in src_text:
            cell["source"] = append_save_call(cell["source"], "residual_analysis.png")
            print("Updated Residual plot cell.")

        elif "MAE by Month" in src_text:
            cell["source"] = append_save_call(cell["source"], "mae_by_month.png")
            print("Updated MAE by Month cell.")

        elif "MAE by Season" in src_text:
            cell["source"] = append_save_call(cell["source"], "mae_by_season.png")
            print("Updated MAE by Season cell.")

        elif "Calibration Curve" in src_text:
             cell["source"] = append_save_call(cell["source"], "calibration_curve.png")
             print("Updated Calibration cell.")

        elif "Error by Lead Time" in src_text or "RMSE by Lead Time" in src_text:
             cell["source"] = append_save_call(cell["source"], "error_by_lead_time.png")
             print("Updated Lead Time Error cell.")

nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Notebook patched with plot export.")
