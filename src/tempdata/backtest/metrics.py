"""Metrics calculation for backtest evaluation.

This module provides:
1. Forecast metrics (MAE, RMSE, bias, interval coverage)
2. Trading metrics (PnL, Sharpe, drawdown, win rate)
3. Sliced metrics by time period, lead hours, temperature regime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ForecastMetrics:
    """Forecast accuracy and calibration metrics.
    
    Attributes:
        n_samples: Number of samples
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        bias: Mean error (predicted - actual)
        coverage_50: Fraction of actuals in 50% prediction interval
        coverage_80: Fraction of actuals in 80% prediction interval
        coverage_90: Fraction of actuals in 90% prediction interval
        brier_score: Brier score on bin probabilities (if available)
    """
    n_samples: int
    mae: float
    rmse: float
    bias: float
    coverage_50: float | None = None
    coverage_80: float | None = None
    coverage_90: float | None = None
    brier_score: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "mae": self.mae,
            "rmse": self.rmse,
            "bias": self.bias,
            "coverage_50": self.coverage_50,
            "coverage_80": self.coverage_80,
            "coverage_90": self.coverage_90,
            "brier_score": self.brier_score,
        }


@dataclass
class TradingMetrics:
    """Trading performance metrics.
    
    Attributes:
        n_trades: Total number of trades
        total_pnl: Total profit/loss
        return_pct: Return as percentage of initial bankroll
        sharpe_ratio: Sharpe ratio (annualized)
        max_drawdown: Maximum drawdown
        max_drawdown_pct: Maximum drawdown as percentage
        win_rate: Fraction of winning trades
        avg_win: Average winning trade P&L
        avg_loss: Average losing trade P&L
        avg_edge: Average edge at time of trade
        avg_edge_captured: Average edge actually captured
        total_volume: Total trading volume
        avg_trade_size: Average trade size
        max_exposure: Maximum exposure reached
    """
    n_trades: int
    total_pnl: float
    return_pct: float
    sharpe_ratio: float | None
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_edge: float
    avg_edge_captured: float
    total_volume: float
    avg_trade_size: float
    max_exposure: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_trades": self.n_trades,
            "total_pnl": self.total_pnl,
            "return_pct": self.return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_edge": self.avg_edge,
            "avg_edge_captured": self.avg_edge_captured,
            "total_volume": self.total_volume,
            "avg_trade_size": self.avg_trade_size,
            "max_exposure": self.max_exposure,
        }


@dataclass
class BacktestMetrics:
    """Complete backtest metrics summary.
    
    Contains both forecast and trading metrics, plus sliced breakdowns.
    """
    forecast: ForecastMetrics
    trading: TradingMetrics
    slices: dict[str, dict] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "forecast": self.forecast.to_dict(),
            "trading": self.trading.to_dict(),
            "slices": self.slices,
        }


def compute_forecast_metrics(
    predictions_df: pd.DataFrame,
    coverage_levels: list[float] | None = None,
) -> ForecastMetrics:
    """Compute forecast accuracy and calibration metrics.
    
    Args:
        predictions_df: DataFrame with mu_f, sigma_f, tmax_actual_f columns
        coverage_levels: Coverage levels to check (default [0.50, 0.80, 0.90])
        
    Returns:
        ForecastMetrics object
    """
    if coverage_levels is None:
        coverage_levels = [0.50, 0.80, 0.90]
    
    # Filter to rows with actual values
    df = predictions_df.dropna(subset=["tmax_actual_f", "mu_f"])
    
    if df.empty:
        return ForecastMetrics(
            n_samples=0,
            mae=np.nan,
            rmse=np.nan,
            bias=np.nan,
        )
    
    actual = df["tmax_actual_f"].values
    predicted = df["mu_f"].values
    
    # Point prediction metrics
    errors = predicted - actual
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    bias = np.mean(errors)
    
    # Interval coverage (if sigma available)
    coverage = {}
    if "sigma_f" in df.columns:
        sigma = df["sigma_f"].values
        
        try:
            from scipy.stats import norm
            
            for level in coverage_levels:
                alpha = (1 - level) / 2
                z = norm.ppf(1 - alpha)
                
                lower = predicted - z * sigma
                upper = predicted + z * sigma
                
                in_interval = (actual >= lower) & (actual <= upper)
                coverage[level] = float(in_interval.mean())
        except ImportError:
            pass
    
    return ForecastMetrics(
        n_samples=len(df),
        mae=float(mae),
        rmse=float(rmse),
        bias=float(bias),
        coverage_50=coverage.get(0.50),
        coverage_80=coverage.get(0.80),
        coverage_90=coverage.get(0.90),
    )


def compute_trading_metrics(
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    initial_bankroll: float,
) -> TradingMetrics:
    """Compute trading performance metrics.
    
    Args:
        trades_df: DataFrame of all trades
        daily_df: DataFrame of daily P&L
        initial_bankroll: Starting bankroll
        
    Returns:
        TradingMetrics object
    """
    # Filter to settled trades
    settled = trades_df[trades_df["is_settled"]].copy()
    
    if settled.empty:
        return TradingMetrics(
            n_trades=0,
            total_pnl=0.0,
            return_pct=0.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_edge=0.0,
            avg_edge_captured=0.0,
            total_volume=0.0,
            avg_trade_size=0.0,
            max_exposure=0.0,
        )
    
    n_trades = len(settled)
    total_pnl = settled["pnl"].sum()
    return_pct = total_pnl / initial_bankroll * 100
    
    # Win/loss stats
    wins = settled[settled["pnl"] > 0]
    losses = settled[settled["pnl"] <= 0]
    
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
    
    # Edge stats
    avg_edge = settled["edge"].mean()
    avg_edge_captured = (settled["pnl"] / settled["size"]).mean() if n_trades > 0 else 0
    
    # Volume stats
    total_volume = settled["size"].sum()
    avg_trade_size = settled["size"].mean()
    
    # Sharpe ratio (if we have daily P&L)
    sharpe_ratio = None
    if not daily_df.empty and "total_pnl" in daily_df.columns:
        daily_pnl = daily_df.groupby("target_date_local")["total_pnl"].sum()
        if len(daily_pnl) > 1:
            mean_daily = daily_pnl.mean()
            std_daily = daily_pnl.std()
            if std_daily > 0:
                sharpe_ratio = float(mean_daily / std_daily * np.sqrt(252))
    
    # Drawdown calculation
    max_drawdown, max_drawdown_pct = _compute_drawdown(daily_df, initial_bankroll)
    
    # Max exposure (approximate from trades)
    max_exposure = _estimate_max_exposure(trades_df)
    
    return TradingMetrics(
        n_trades=n_trades,
        total_pnl=float(total_pnl),
        return_pct=float(return_pct),
        sharpe_ratio=sharpe_ratio,
        max_drawdown=float(max_drawdown),
        max_drawdown_pct=float(max_drawdown_pct),
        win_rate=float(win_rate),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        avg_edge=float(avg_edge),
        avg_edge_captured=float(avg_edge_captured),
        total_volume=float(total_volume),
        avg_trade_size=float(avg_trade_size),
        max_exposure=float(max_exposure),
    )


def _compute_drawdown(
    daily_df: pd.DataFrame,
    initial_bankroll: float,
) -> tuple[float, float]:
    """Compute maximum drawdown from daily results."""
    if daily_df.empty or "total_pnl" not in daily_df.columns:
        return 0.0, 0.0
    
    # Aggregate by date
    daily_pnl = daily_df.groupby("target_date_local")["total_pnl"].sum().sort_index()
    
    if len(daily_pnl) == 0:
        return 0.0, 0.0
    
    # Compute cumulative P&L
    cumulative = daily_pnl.cumsum()
    
    # Compute running maximum
    running_max = cumulative.cummax()
    
    # Drawdown = running_max - current
    drawdown = running_max - cumulative
    
    max_drawdown = drawdown.max()
    max_drawdown_pct = max_drawdown / initial_bankroll * 100 if initial_bankroll > 0 else 0
    
    return max_drawdown, max_drawdown_pct


def _estimate_max_exposure(trades_df: pd.DataFrame) -> float:
    """Estimate maximum exposure from trades."""
    if trades_df.empty:
        return 0.0
    
    # This is an approximation - in reality, exposure depends on timing
    # Here we use sum of concurrent exposure by date
    by_date = trades_df.groupby("target_date_local")["size"].sum()
    return float(by_date.max()) if len(by_date) > 0 else 0.0


def compute_metrics_by_slice(
    predictions_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    initial_bankroll: float,
    df_full: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Compute metrics broken down by various slices.
    
    Slices:
    - By month
    - By lead_hours bucket
    - By temperature regime (cold/normal/hot)
    
    Args:
        predictions_df: Predictions with forecasts
        trades_df: All trades
        daily_df: Daily P&L
        initial_bankroll: Starting bankroll
        df_full: Full data for additional context
        
    Returns:
        Dictionary of slice name -> metrics dict
    """
    slices = {}
    
    # By month
    if "target_date_local" in predictions_df.columns:
        predictions_df = predictions_df.copy()
        predictions_df["month"] = pd.to_datetime(
            predictions_df["target_date_local"]
        ).dt.month
        
        month_metrics = {}
        for month in range(1, 13):
            month_preds = predictions_df[predictions_df["month"] == month]
            if len(month_preds) > 0:
                fm = compute_forecast_metrics(month_preds)
                month_metrics[month] = {
                    "n_samples": fm.n_samples,
                    "mae": fm.mae,
                    "rmse": fm.rmse,
                    "bias": fm.bias,
                }
        slices["by_month"] = month_metrics
    
    # By lead_hours bucket (if available)
    if df_full is not None and "lead_hours" in df_full.columns:
        lead_buckets = [(0, 24), (24, 48), (48, 72), (72, 168), (168, 720)]
        lead_metrics = {}
        
        for lo, hi in lead_buckets:
            mask = (df_full["lead_hours"] >= lo) & (df_full["lead_hours"] < hi)
            indices = df_full[mask].index
            
            bucket_preds = predictions_df[predictions_df.index.isin(indices)]
            if len(bucket_preds) > 0:
                fm = compute_forecast_metrics(bucket_preds)
                lead_metrics[f"{lo}-{hi}h"] = {
                    "n_samples": fm.n_samples,
                    "mae": fm.mae,
                    "rmse": fm.rmse,
                }
        slices["by_lead_hours"] = lead_metrics
    
    # By temperature regime
    if "tmax_actual_f" in predictions_df.columns:
        actuals = predictions_df["tmax_actual_f"].dropna()
        if len(actuals) > 0:
            q25 = actuals.quantile(0.25)
            q75 = actuals.quantile(0.75)
            
            regime_metrics = {}
            
            # Cold
            cold_mask = predictions_df["tmax_actual_f"] < q25
            cold_preds = predictions_df[cold_mask]
            if len(cold_preds) > 0:
                fm = compute_forecast_metrics(cold_preds)
                regime_metrics["cold"] = {"n_samples": fm.n_samples, "mae": fm.mae, "bias": fm.bias}
            
            # Normal
            normal_mask = (predictions_df["tmax_actual_f"] >= q25) & (predictions_df["tmax_actual_f"] <= q75)
            normal_preds = predictions_df[normal_mask]
            if len(normal_preds) > 0:
                fm = compute_forecast_metrics(normal_preds)
                regime_metrics["normal"] = {"n_samples": fm.n_samples, "mae": fm.mae, "bias": fm.bias}
            
            # Hot
            hot_mask = predictions_df["tmax_actual_f"] > q75
            hot_preds = predictions_df[hot_mask]
            if len(hot_preds) > 0:
                fm = compute_forecast_metrics(hot_preds)
                regime_metrics["hot"] = {"n_samples": fm.n_samples, "mae": fm.mae, "bias": fm.bias}
            
            slices["by_temp_regime"] = regime_metrics
    
    # Trading by month
    if not trades_df.empty and "target_date_local" in trades_df.columns:
        trades_df = trades_df.copy()
        trades_df["month"] = pd.to_datetime(trades_df["target_date_local"]).dt.month
        
        trade_month_metrics = {}
        for month in range(1, 13):
            month_trades = trades_df[trades_df["month"] == month]
            settled = month_trades[month_trades["is_settled"]]
            if len(settled) > 0:
                trade_month_metrics[month] = {
                    "n_trades": len(settled),
                    "total_pnl": float(settled["pnl"].sum()),
                    "win_rate": float((settled["pnl"] > 0).mean()),
                }
        slices["trading_by_month"] = trade_month_metrics
    
    return slices


def compute_all_metrics(
    predictions_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    initial_bankroll: float,
    df_full: pd.DataFrame | None = None,
) -> BacktestMetrics:
    """Compute all backtest metrics.
    
    Args:
        predictions_df: Predictions DataFrame
        trades_df: Trades DataFrame
        daily_df: Daily results DataFrame
        initial_bankroll: Starting bankroll
        df_full: Full dataset for slicing context
        
    Returns:
        BacktestMetrics with all metrics
    """
    forecast = compute_forecast_metrics(predictions_df)
    trading = compute_trading_metrics(trades_df, daily_df, initial_bankroll)
    slices = compute_metrics_by_slice(
        predictions_df, trades_df, daily_df, initial_bankroll, df_full
    )
    
    return BacktestMetrics(
        forecast=forecast,
        trading=trading,
        slices=slices,
    )


def print_metrics_summary(metrics: BacktestMetrics) -> None:
    """Print a formatted summary of backtest metrics."""
    print("\n" + "=" * 60)
    print("BACKTEST METRICS SUMMARY")
    print("=" * 60)
    
    print("\n--- FORECAST METRICS ---")
    fm = metrics.forecast
    print(f"  Samples:     {fm.n_samples:,}")
    print(f"  MAE:         {fm.mae:.2f}°F")
    print(f"  RMSE:        {fm.rmse:.2f}°F")
    print(f"  Bias:        {fm.bias:+.2f}°F")
    if fm.coverage_50 is not None:
        print(f"  50% PI:      {fm.coverage_50*100:.1f}% coverage")
    if fm.coverage_80 is not None:
        print(f"  80% PI:      {fm.coverage_80*100:.1f}% coverage")
    if fm.coverage_90 is not None:
        print(f"  90% PI:      {fm.coverage_90*100:.1f}% coverage")
    
    print("\n--- TRADING METRICS ---")
    tm = metrics.trading
    print(f"  Trades:      {tm.n_trades:,}")
    print(f"  Total PnL:   ${tm.total_pnl:,.2f}")
    print(f"  Return:      {tm.return_pct:+.2f}%")
    if tm.sharpe_ratio is not None:
        print(f"  Sharpe:      {tm.sharpe_ratio:.2f}")
    print(f"  Max DD:      ${tm.max_drawdown:,.2f} ({tm.max_drawdown_pct:.1f}%)")
    print(f"  Win Rate:    {tm.win_rate*100:.1f}%")
    print(f"  Avg Win:     ${tm.avg_win:,.2f}")
    print(f"  Avg Loss:    ${tm.avg_loss:,.2f}")
    print(f"  Avg Edge:    {tm.avg_edge*100:.1f}%")
    print(f"  Volume:      ${tm.total_volume:,.2f}")
    
    print("=" * 60 + "\n")
