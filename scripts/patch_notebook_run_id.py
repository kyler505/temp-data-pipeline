
import json
import sys
from pathlib import Path

def patch_notebook():
    nb_path = Path("notebooks/backtest_analysis.ipynb")
    if not nb_path.exists():
        print(f"Notebook not found: {nb_path}")
        sys.exit(1)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    patched = False

    # Iterate through cells to find the code cell with the hardcoded RUN_ID
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            new_source = []
            changed_cell = False

            for line in source:
                # Look for the specific line we identified
                if 'RUN_ID = "bt_klga_baseline_2025_01_19"' in line:
                    # Replace with None
                    new_line = line.replace('"bt_klga_baseline_2025_01_19"', 'None')
                    # Update comment to be more accurate if needed, or just leave it
                    # The original line was: RUN_ID = "..." # Or None for timestamp-based ID
                    # The new line will be: RUN_ID = None # Or None...
                    # Let's clean it up slightly to: RUN_ID = None  # Auto-generated
                    new_line = 'RUN_ID = None  # Auto-generated timestamp-based ID\n'
                    new_source.append(new_line)
                    changed_cell = True
                    patched = True
                    print("Found and patched RUN_ID line.")
                else:
                    new_source.append(line)

            if changed_cell:
                cell["source"] = new_source

    if patched:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully patched {nb_path}")
    else:
        print("Could not find the target line to patch.")

if __name__ == "__main__":
    patch_notebook()
