#!/usr/bin/env python3
"""CLI for generating evaluation reports and plots.

Usage:
    python scripts/generate_report.py --run-id <run_id>
"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tempdata.eval.runner import load_run, load_multi_model_run, list_runs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation report.")
    parser.add_argument(
        "--run-id",
        help="Run identifier (default: latest)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Base directory for runs (default: runs)",
    )
    return parser.parse_args()

def save_plot(fig, filename, run_id, base_dir):
    """Save figure to the run's plot directory."""
    plot_dir = base_dir / run_id / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    out_path = plot_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {out_path}")
    plt.close(fig)

def plot_residuals(results, run_id, base_dir):
    if not results: return

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        if 'residuals_df' in data and data['residuals_df'] is not None:
            res = data['residuals_df']['residual_f']
            sns.kdeplot(res, label=f"{name} (Mean: {res.mean():.2f})", fill=True, alpha=0.1, ax=ax)

    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_title("Residual Distribution Comparison")
    ax.set_xlabel("Residual (Forecast - Truth)")
    ax.legend()
    save_plot(fig, "residuals_comparison.png", run_id, base_dir)

def plot_mae_by_month(results, run_id, base_dir):
    monthly_data = []
    for name, data in results.items():
        if data.get('slices') and 'by_month' in data['slices']:
            by_month = data['slices']['by_month']
            for m, vals in by_month.items():
                monthly_data.append({
                    "Model": name,
                    "Month": int(m),
                    "MAE": vals['mae']
                })

    if monthly_data:
        df_m = pd.DataFrame(monthly_data)
        fig = plt.figure(figsize=(12, 6))
        sns.barplot(data=df_m, x="Month", y="MAE", hue="Model")
        plt.title("MAE by Month Comparison")
        plt.grid(axis='y', alpha=0.3)
        save_plot(fig, "mae_by_month_comparison.png", run_id, base_dir)

def plot_mae_by_season(results, run_id, base_dir):
    season_data = []
    for name, data in results.items():
        if data.get('slices') and 'by_season' in data['slices']:
            by_season = data['slices']['by_season']
            for s, vals in by_season.items():
                season_data.append({
                    "Model": name,
                    "Season": s,
                    "MAE": vals['mae']
                })

    if season_data:
        df_s = pd.DataFrame(season_data)
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(data=df_s, x="Season", y="MAE", hue="Model")
        plt.title("MAE by Season Comparison")
        plt.grid(axis='y', alpha=0.3)
        save_plot(fig, "mae_by_season_comparison.png", run_id, base_dir)

def main():
    args = parse_args()

    # 1. Load Run Data
    if not args.run_id:
        runs = list_runs(args.output_dir)
        if not runs:
            print("No runs found.")
            return
        run_id = runs[0]['run_id']
        print(f"Using latest run: {run_id}")
    else:
        run_id = args.run_id

    print(f"Loading run {run_id}...")
    try:
        # Check if multi-model (hacky way: try loading as multi, fallback to single)
        # Actually list_runs gives us a hint, but let's just try loading
        run_path = args.output_dir / run_id
        if (run_path / "multi_model_config.json").exists():
             multi_data = load_multi_model_run(run_id, args.output_dir)
             results = multi_data['models']
        else:
             data = load_run(run_id, args.output_dir)
             name = data.get('config').run_name if data.get('config') else 'Model'
             results = {name: data}

        print(f"Loaded models: {list(results.keys())}")
    except Exception as e:
        print(f"Error loading run: {e}")
        return

    # 2. Generate Plots
    print("Generating plots...")
    plot_residuals(results, run_id, args.output_dir)
    plot_mae_by_month(results, run_id, args.output_dir)
    plot_mae_by_season(results, run_id, args.output_dir)

    print("Done.")

if __name__ == "__main__":
    main()
