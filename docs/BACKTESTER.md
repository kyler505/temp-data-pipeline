# Temperature Backtester

A complete backtesting framework for evaluating daily Tmax predictions and trading strategies on Polymarket-style markets.

## Overview

The backtester evaluates the full pipeline from forecasts to trading P&L:

```
Forecasts → Features → Predictions → Bin Probabilities → Market Prices → Orders → Trades → P&L
```

## Quick Start

### CLI Usage

```bash
# Install dependencies
pip install -e ".[backtest]"

# Run a backtest
python scripts/backtest_daily_tmax.py \
    --station KLGA \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --run-id my_experiment
```

### Programmatic Usage

```python
from tempdata.backtest import (
    BacktestConfig, load_backtest_data, run_backtest,
    create_forecaster, create_uncertainty_model,
    create_strategy, create_price_provider, BinSet
)

# Configure
config = BacktestConfig(
    station_ids=["KLGA"],
    start_date_local=date(2020, 1, 1),
    end_date_local=date(2024, 12, 31),
)

# Load data
dataset = load_backtest_data(config, forecast_df, truth_df)

# Run backtest
trades_df, daily_df, predictions_df = run_backtest(
    df_train=dataset.train,
    df_test=dataset.test,
    forecaster=create_forecaster("ridge"),
    uncertainty_model=create_uncertainty_model("bucketed"),
    strategy=create_strategy("edge"),
    price_provider=create_price_provider("synthetic"),
    bins=BinSet(config.bins_f),
)
```

## Components

### Data Pipeline
- **Data Loading**: Joins forecasts to truth data with QC filtering
- **Feature Engineering**: Seasonal features, rolling bias/error statistics
- **Time Splitting**: Train/val/test splits respecting temporal order

### Models
- **Forecaster**: Point predictions (mu) with Ridge regression baseline
- **Uncertainty Model**: Sigma estimation (global, bucketed, or rolling)

### Trading
- **Bin Probabilities**: Normal CDF conversion to market probabilities
- **Price Providers**: Synthetic, fixed spread, or adversarial pricing
- **Strategies**: Edge-based, Kelly criterion, or no-op

### Simulation
- **Order Execution**: With slippage and market impact
- **Settlement**: P&L calculation on market resolution
- **Risk Management**: Position limits and exposure controls

### Metrics
- **Forecast**: MAE, RMSE, bias, interval coverage
- **Trading**: P&L, Sharpe ratio, win rate, drawdown
- **Slicing**: By month, lead hours, temperature regime

## Output Artifacts

Each run creates a directory `runs/<run_id>/` with:

- `config.json`: Frozen configuration
- `metrics.json`: Complete metrics summary
- `trades.parquet`: All executed trades
- `daily_results.parquet`: Daily P&L summary
- `predictions.parquet`: Model predictions and probabilities
- `summary.txt`: Human-readable report

## Configuration

Key parameters in `BacktestConfig`:

```python
config = BacktestConfig(
    # Data
    station_ids=["KLGA"],
    start_date_local=date(2020, 1, 1),
    end_date_local=date(2024, 12, 31),
    lead_hours_allowed=[28, 29],

    # Model
    model_type="ridge",  # "ridge" or "passthrough"
    sigma_type="bucketed",  # "global", "bucketed", or "rolling"

    # Trading
    edge_min=0.03,  # Minimum edge to trade
    max_per_market_pct=0.02,  # Max size per market
    max_total_pct=0.25,  # Max total exposure

    # Market
    bins_f=[(float("-inf"), 50.0), (50.0, 70.0), (70.0, float("inf"))],
)
```

## Example Output

```
BACKTEST METRICS SUMMARY
============================================================

--- FORECAST PERFORMANCE ---
  Samples:      1,000
  MAE:          3.45°F
  RMSE:         4.32°F
  Bias:         -1.23°F
  90% PI cov:   87.3%

--- TRADING PERFORMANCE ---
  Total trades: 456
  Total PnL:    $2,341.50
  Return:       23.4%
  Sharpe:       1.67
  Max DD:       $1,234.50 (12.3%)
  Win rate:     58.1%
  Avg edge:     2.3%
```

## Architecture

```
src/tempdata/backtest/
├── config.py          # Configuration management
├── data.py            # Data loading and joining
├── models.py          # Forecast models
├── calibration.py     # Uncertainty models
├── bins.py            # Bin definitions and probabilities
├── pricing.py         # Market price providers
├── strategy.py        # Trading strategies
├── simulator.py       # Trade execution
├── metrics.py         # Performance metrics
├── report.py          # Artifact generation
└── __init__.py        # Public API
```

## Testing

Run the test suite:

```bash
pytest tests/test_backtest.py -v
```

Run the example script:

```bash
python scripts/example_backtest.py
```

## Common Patterns

### Debugging Overfitting

```python
# Use simple models first
forecaster = create_forecaster("passthrough")
uncertainty_model = create_uncertainty_model("global")

# Check interval coverage
metrics.forecast.coverage_90  # Should be ~90%
```

### Market Realism

```python
# Add market friction
strategy = create_strategy("edge", slippage=0.02)

# Use adversarial pricing for stress testing
price_provider = create_price_provider("adversarial", market_edge=0.2)
```

### Performance Analysis

```python
# Check by lead time bucket
lead_metrics = metrics.slices.get("by_lead_hours", {})
print("MAE by lead time:", {
    bucket: data["mae"] for bucket, data in lead_metrics.items()
})
```