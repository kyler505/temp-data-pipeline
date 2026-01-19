"""Backtesting framework for daily Tmax prediction and trading strategies.

This module provides a complete backtesting pipeline:
    forecasts → features → prediction distribution → bin probabilities → trades → PnL

Key components:
    - BacktestConfig: Configuration for backtest runs
    - Forecaster: Model interface for point predictions
    - UncertaintyModel: Interface for calibrated uncertainty
    - Strategy: Trading strategy interface
    - Simulator: Trade execution and settlement

Example usage:
    from tempdata.backtest import (
        BacktestConfig,
        load_backtest_data,
        run_backtest,
        create_forecaster,
        create_uncertainty_model,
        create_strategy,
        create_price_provider,
        BinSet,
    )
    
    config = BacktestConfig(
        station_ids=["KLGA"],
        start_date_local=date(2020, 1, 1),
        end_date_local=date(2024, 12, 31),
    )
    
    dataset = load_backtest_data(config, forecast_df, truth_df)
    
    trades_df, daily_df, predictions_df = run_backtest(
        df_train=dataset.train,
        df_test=dataset.test,
        forecaster=create_forecaster("ridge"),
        uncertainty_model=create_uncertainty_model("bucketed"),
        strategy=create_strategy("edge"),
        price_provider=create_price_provider("synthetic"),
        bins=BinSet(config.bins_f),
    )
"""

from tempdata.backtest.bins import BinSet, compute_bin_probabilities
from tempdata.backtest.calibration import (
    BucketedSigma,
    GlobalSigma,
    RollingSigma,
    UncertaintyModel,
    create_uncertainty_model,
)
from tempdata.backtest.config import BacktestConfig, generate_run_id
from tempdata.backtest.data import (
    BacktestDataset,
    check_data_overlap,
    load_backtest_data,
    print_data_summary,
)
from tempdata.backtest.metrics import (
    BacktestMetrics,
    ForecastMetrics,
    TradingMetrics,
    compute_all_metrics,
    compute_forecast_metrics,
    compute_trading_metrics,
    print_metrics_summary,
)
from tempdata.backtest.models import (
    Forecaster,
    PassthroughForecaster,
    RidgeForecaster,
    create_forecaster,
)
from tempdata.backtest.pricing import (
    FixedSpreadPriceProvider,
    PriceProvider,
    SyntheticPriceProvider,
    create_price_provider,
)
from tempdata.backtest.report import (
    create_run_dir,
    list_runs,
    load_run,
    write_all_artifacts,
)
from tempdata.backtest.simulator import Simulator, run_backtest
from tempdata.backtest.strategy import (
    EdgeStrategy,
    KellyStrategy,
    Order,
    Side,
    Strategy,
    create_strategy,
)

__all__ = [
    # Config
    "BacktestConfig",
    "generate_run_id",
    # Data
    "BacktestDataset",
    "load_backtest_data",
    "check_data_overlap",
    "print_data_summary",
    # Models
    "Forecaster",
    "RidgeForecaster",
    "PassthroughForecaster",
    "create_forecaster",
    # Calibration
    "UncertaintyModel",
    "GlobalSigma",
    "BucketedSigma",
    "RollingSigma",
    "create_uncertainty_model",
    # Bins
    "BinSet",
    "compute_bin_probabilities",
    # Pricing
    "PriceProvider",
    "SyntheticPriceProvider",
    "FixedSpreadPriceProvider",
    "create_price_provider",
    # Strategy
    "Strategy",
    "EdgeStrategy",
    "KellyStrategy",
    "Order",
    "Side",
    "create_strategy",
    # Simulator
    "Simulator",
    "run_backtest",
    # Metrics
    "ForecastMetrics",
    "TradingMetrics",
    "BacktestMetrics",
    "compute_forecast_metrics",
    "compute_trading_metrics",
    "compute_all_metrics",
    "print_metrics_summary",
    # Report
    "create_run_dir",
    "write_all_artifacts",
    "load_run",
    "list_runs",
]
