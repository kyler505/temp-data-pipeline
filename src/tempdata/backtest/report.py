"""Report generation and artifact writing for backtest runs.

This module handles:
1. Creating run directories
2. Writing all artifacts (config, metrics, DataFrames)
3. Optional visualization generation
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tempdata.backtest.config import BacktestConfig
    from tempdata.backtest.metrics import BacktestMetrics


def get_runs_dir(base_path: Path | str | None = None) -> Path:
    """Get the runs directory path.
    
    Args:
        base_path: Base path for runs. If None, uses project root.
        
    Returns:
        Path to runs directory
    """
    if base_path is not None:
        return Path(base_path) / "runs"
    
    # Try to find project root
    try:
        from tempdata.config import project_root
        return project_root() / "runs"
    except ImportError:
        return Path("runs")


def create_run_dir(
    run_id: str,
    base_path: Path | str | None = None,
) -> Path:
    """Create a run directory.
    
    Args:
        run_id: Unique identifier for this run
        base_path: Base path for runs directory
        
    Returns:
        Path to the created run directory
    """
    runs_dir = get_runs_dir(base_path)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_config(
    config: BacktestConfig,
    run_dir: Path,
) -> Path:
    """Write configuration to JSON file.
    
    Args:
        config: Backtest configuration
        run_dir: Run directory path
        
    Returns:
        Path to written config file
    """
    config_path = run_dir / "config.json"
    config.save(config_path)
    return config_path


def write_metrics(
    metrics: BacktestMetrics,
    run_dir: Path,
) -> Path:
    """Write metrics to JSON file.
    
    Args:
        metrics: Backtest metrics
        run_dir: Run directory path
        
    Returns:
        Path to written metrics file
    """
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2, default=str))
    return metrics_path


def write_trades(
    trades_df: pd.DataFrame,
    run_dir: Path,
) -> Path:
    """Write trades DataFrame to parquet.
    
    Args:
        trades_df: Trades DataFrame
        run_dir: Run directory path
        
    Returns:
        Path to written parquet file
    """
    trades_path = run_dir / "trades.parquet"
    trades_df.to_parquet(trades_path, index=False)
    return trades_path


def write_daily_results(
    daily_df: pd.DataFrame,
    run_dir: Path,
) -> Path:
    """Write daily results DataFrame to parquet.
    
    Args:
        daily_df: Daily results DataFrame
        run_dir: Run directory path
        
    Returns:
        Path to written parquet file
    """
    daily_path = run_dir / "daily_results.parquet"
    daily_df.to_parquet(daily_path, index=False)
    return daily_path


def write_predictions(
    predictions_df: pd.DataFrame,
    run_dir: Path,
) -> Path:
    """Write predictions DataFrame to parquet.
    
    Args:
        predictions_df: Predictions DataFrame
        run_dir: Run directory path
        
    Returns:
        Path to written parquet file
    """
    predictions_path = run_dir / "predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    return predictions_path


def write_all_artifacts(
    config: BacktestConfig,
    metrics: BacktestMetrics,
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    run_id: str,
    base_path: Path | str | None = None,
) -> dict[str, Path]:
    """Write all run artifacts.
    
    Args:
        config: Backtest configuration
        metrics: Backtest metrics
        trades_df: Trades DataFrame
        daily_df: Daily results DataFrame
        predictions_df: Predictions DataFrame
        run_id: Run identifier
        base_path: Base path for runs directory
        
    Returns:
        Dictionary mapping artifact name to path
    """
    run_dir = create_run_dir(run_id, base_path)
    
    artifacts = {
        "config": write_config(config, run_dir),
        "metrics": write_metrics(metrics, run_dir),
        "trades": write_trades(trades_df, run_dir),
        "daily_results": write_daily_results(daily_df, run_dir),
        "predictions": write_predictions(predictions_df, run_dir),
    }
    
    # Write run manifest
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    artifacts["manifest"] = manifest_path
    
    print(f"[report] Wrote artifacts to {run_dir}")
    
    return artifacts


def load_run(
    run_id: str,
    base_path: Path | str | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Load a previous run's artifacts.
    
    Args:
        run_id: Run identifier
        base_path: Base path for runs directory
        
    Returns:
        Dictionary with loaded artifacts
    """
    from tempdata.backtest.config import BacktestConfig
    
    runs_dir = get_runs_dir(base_path)
    run_dir = runs_dir / run_id
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    result = {}
    
    # Load config
    config_path = run_dir / "config.json"
    if config_path.exists():
        result["config"] = BacktestConfig.load(config_path)
    
    # Load metrics
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        result["metrics"] = json.loads(metrics_path.read_text())
    
    # Load DataFrames
    trades_path = run_dir / "trades.parquet"
    if trades_path.exists():
        result["trades"] = pd.read_parquet(trades_path)
    
    daily_path = run_dir / "daily_results.parquet"
    if daily_path.exists():
        result["daily_results"] = pd.read_parquet(daily_path)
    
    predictions_path = run_dir / "predictions.parquet"
    if predictions_path.exists():
        result["predictions"] = pd.read_parquet(predictions_path)
    
    return result


def list_runs(
    base_path: Path | str | None = None,
) -> list[dict]:
    """List all available runs.
    
    Args:
        base_path: Base path for runs directory
        
    Returns:
        List of run info dictionaries
    """
    runs_dir = get_runs_dir(base_path)
    
    if not runs_dir.exists():
        return []
    
    runs = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                runs.append(manifest)
            else:
                # Create basic info from directory
                runs.append({
                    "run_id": run_dir.name,
                    "created_at": None,
                })
    
    # Sort by creation time
    runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    
    return runs


def generate_summary_report(
    config: BacktestConfig,
    metrics: BacktestMetrics,
    run_id: str,
) -> str:
    """Generate a text summary report.
    
    Args:
        config: Backtest configuration
        metrics: Backtest metrics
        run_id: Run identifier
        
    Returns:
        Formatted text report
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"BACKTEST REPORT: {run_id}")
    lines.append("=" * 70)
    
    # Configuration summary
    lines.append("\n--- CONFIGURATION ---")
    lines.append(f"  Stations:     {', '.join(config.station_ids)}")
    lines.append(f"  Date range:   {config.start_date_local} to {config.end_date_local}")
    if config.lead_hours_allowed:
        lines.append(f"  Lead hours:   {config.lead_hours_allowed}")
    lines.append(f"  Model:        {config.model_type} (alpha={config.model_alpha})")
    lines.append(f"  Edge min:     {config.edge_min*100:.1f}%")
    lines.append(f"  Initial $:    ${config.initial_bankroll:,.0f}")
    
    # Forecast metrics
    lines.append("\n--- FORECAST PERFORMANCE ---")
    fm = metrics.forecast
    lines.append(f"  Samples:      {fm.n_samples:,}")
    lines.append(f"  MAE:          {fm.mae:.2f}°F")
    lines.append(f"  RMSE:         {fm.rmse:.2f}°F")
    lines.append(f"  Bias:         {fm.bias:+.2f}°F")
    if fm.coverage_90:
        lines.append(f"  90% PI cov:   {fm.coverage_90*100:.1f}%")
    
    # Trading metrics
    lines.append("\n--- TRADING PERFORMANCE ---")
    tm = metrics.trading
    lines.append(f"  Total trades: {tm.n_trades:,}")
    lines.append(f"  Total PnL:    ${tm.total_pnl:+,.2f}")
    lines.append(f"  Return:       {tm.return_pct:+.2f}%")
    if tm.sharpe_ratio is not None:
        lines.append(f"  Sharpe:       {tm.sharpe_ratio:.2f}")
    lines.append(f"  Max DD:       ${tm.max_drawdown:,.2f} ({tm.max_drawdown_pct:.1f}%)")
    lines.append(f"  Win rate:     {tm.win_rate*100:.1f}%")
    lines.append(f"  Avg edge:     {tm.avg_edge*100:.2f}%")
    
    # Sliced metrics summary
    if "by_month" in metrics.slices:
        lines.append("\n--- FORECAST MAE BY MONTH ---")
        for month, data in sorted(metrics.slices["by_month"].items()):
            month_name = datetime(2000, int(month), 1).strftime("%b")
            lines.append(f"  {month_name}: {data['mae']:.2f}°F (n={data['n_samples']})")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


def write_summary_report(
    config: BacktestConfig,
    metrics: BacktestMetrics,
    run_id: str,
    run_dir: Path,
) -> Path:
    """Write summary report to text file.
    
    Args:
        config: Backtest configuration
        metrics: Backtest metrics
        run_id: Run identifier
        run_dir: Run directory path
        
    Returns:
        Path to written report file
    """
    report = generate_summary_report(config, metrics, run_id)
    report_path = run_dir / "summary.txt"
    report_path.write_text(report)
    return report_path
