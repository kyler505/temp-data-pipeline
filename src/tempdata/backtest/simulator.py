"""Trade execution simulation and settlement.

This module provides:
1. Simulator class for running backtests
2. Trade execution with slippage
3. Settlement on market resolution
4. Output DataFrames for analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tempdata.backtest.bins import BinSet, compute_bin_probabilities
from tempdata.backtest.pricing import MarketPrices
from tempdata.backtest.strategy import Order, Side

if TYPE_CHECKING:
    from tempdata.backtest.calibration import UncertaintyModel
    from tempdata.backtest.models import Forecaster
    from tempdata.backtest.pricing import PriceProvider
    from tempdata.backtest.strategy import Strategy


@dataclass
class Trade:
    """A filled trade record.
    
    Attributes:
        trade_id: Unique identifier
        row_idx: Index of the data row
        station_id: Station identifier
        issue_time_utc: When forecast was issued
        target_date_local: Target date being predicted
        bin_id: Which bin was traded
        side: Buy or sell
        size: Trade size in dollars
        fill_price: Actual fill price (including slippage)
        edge: Expected edge at time of trade
        model_prob: Model's probability
        market_price: Market price before slippage
        actual_value: Actual temperature (if known)
        winning_bin: Which bin won (if resolved)
        pnl: Realized P&L (if resolved)
        is_settled: Whether trade has been settled
    """
    trade_id: int
    row_idx: int
    station_id: str
    issue_time_utc: pd.Timestamp
    target_date_local: pd.Timestamp
    bin_id: int
    side: str
    size: float
    fill_price: float
    edge: float
    model_prob: float
    market_price: float
    actual_value: float | None = None
    winning_bin: int | None = None
    pnl: float | None = None
    is_settled: bool = False
    reserved_margin: float = 0.0


@dataclass
class SimulatorState:
    """Internal state for the simulator.
    
    Tracks bankroll, exposure, and trade history during simulation.
    """
    bankroll: float
    initial_bankroll: float
    exposure: dict[int, float] = field(default_factory=dict)  # bin_id -> exposure
    trades: list[Trade] = field(default_factory=list)
    trade_counter: int = 0
    reserved_margin: float = 0.0
    
    def get_total_exposure(self) -> float:
        """Get total absolute exposure across all bins."""
        return sum(abs(e) for e in self.exposure.values())
    
    def add_exposure(self, bin_id: int, amount: float) -> None:
        """Add exposure to a bin."""
        self.exposure[bin_id] = self.exposure.get(bin_id, 0) + amount
    
    def next_trade_id(self) -> int:
        """Get next trade ID and increment counter."""
        self.trade_counter += 1
        return self.trade_counter


class Simulator:
    """Trade execution simulator for backtesting.
    
    Runs the full simulation loop:
    1. For each row, generate model predictions
    2. Get market prices
    3. Generate orders from strategy
    4. Execute orders with slippage
    5. Settle on resolution
    
    Args:
        forecaster: Model for point predictions
        uncertainty_model: Model for sigma estimation
        strategy: Trading strategy
        price_provider: Market price generator
        bins: Bin definitions
        slippage: Execution slippage (default 0.01)
        initial_bankroll: Starting bankroll (default 10000.0)
    """
    
    def __init__(
        self,
        forecaster: Forecaster,
        uncertainty_model: UncertaintyModel,
        strategy: Strategy,
        price_provider: PriceProvider,
        bins: BinSet,
        slippage: float = 0.01,
        initial_bankroll: float = 10000.0,
    ) -> None:
        self.forecaster = forecaster
        self.uncertainty_model = uncertainty_model
        self.strategy = strategy
        self.price_provider = price_provider
        self.bins = bins
        self.slippage = slippage
        self.initial_bankroll = initial_bankroll
    
    def run(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run the simulation on the given data.
        
        Args:
            df: DataFrame with backtest rows (must have predictions if model not fitted)
            verbose: Whether to print progress
            
        Returns:
            Tuple of (trades_df, daily_results_df, predictions_df)
        """
        if df.empty:
            return self._empty_results()
        
        # Initialize state
        state = SimulatorState(
            bankroll=self.initial_bankroll,
            initial_bankroll=self.initial_bankroll,
        )
        
        # Generate predictions for all rows
        if verbose:
            print("[simulator] Generating predictions...")
        mu = self.forecaster.predict_mu(df)
        sigma = self.uncertainty_model.predict_sigma(df)
        bin_probs = compute_bin_probabilities(mu, sigma, self.bins)
        
        # Store predictions
        predictions = []
        
        # Sort by target_date for proper chronological processing
        df = df.sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)
        
        # Process each row
        if verbose:
            print(f"[simulator] Processing {len(df)} rows...")
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            row_probs = bin_probs[idx]
            row_mu = mu[idx]
            row_sigma = sigma[idx]
            
            # Store prediction
            pred_record = {
                "row_idx": idx,
                "station_id": row["station_id"],
                "issue_time_utc": row["issue_time_utc"],
                "target_date_local": row["target_date_local"],
                "mu_f": row_mu,
                "sigma_f": row_sigma,
                "tmax_actual_f": row.get("tmax_actual_f"),
            }
            # Add bin probabilities
            for bin_id in range(len(self.bins)):
                pred_record[f"p_bin_{bin_id}"] = row_probs[bin_id]
            predictions.append(pred_record)
            
            # Get market prices
            prices = self.price_provider.get_prices(row, row_probs)
            
            # Generate orders
            orders = self.strategy.generate_orders(
                row=row,
                prices=prices,
                bin_probs=row_probs,
                bankroll=state.bankroll,
                exposure=state.exposure.copy(),
            )
            
            # Execute orders
            for order in orders:
                self._execute_order(order, row, idx, state, prices)
        
        # Settle all trades
        if verbose:
            print("[simulator] Settling trades...")
        self._settle_trades(state, df)
        
        # Build output DataFrames
        trades_df = self._build_trades_df(state)
        daily_df = self._build_daily_results_df(state, df)
        predictions_df = pd.DataFrame(predictions)
        
        if verbose:
            print(f"[simulator] Complete: {len(state.trades)} trades")
        
        return trades_df, daily_df, predictions_df
    
    def _execute_order(
        self,
        order: Order,
        row: pd.Series,
        row_idx: int,
        state: SimulatorState,
        prices: MarketPrices,
    ) -> None:
        """Execute a single order with slippage."""
        # Check bid/ask and apply slippage
        if order.side == Side.BUY:
            ask = prices.ask.get(order.bin_id)
            if ask is None or order.limit_price < ask:
                return

            fill_price = min(ask + self.slippage, 0.99)
            cost = order.size * fill_price
            if state.bankroll - state.reserved_margin < cost:
                return
        else:
            bid = prices.bid.get(order.bin_id)
            if bid is None or order.limit_price > bid:
                return

            fill_price = max(bid - self.slippage, 0.01)
            required_margin = order.size * (1 - fill_price)
            if state.bankroll - state.reserved_margin < required_margin:
                return
        
        # Create trade record
        trade = Trade(
            trade_id=state.next_trade_id(),
            row_idx=row_idx,
            station_id=row["station_id"],
            issue_time_utc=row["issue_time_utc"],
            target_date_local=row["target_date_local"],
            bin_id=order.bin_id,
            side=order.side.value,
            size=order.size,
            fill_price=fill_price,
            edge=order.edge,
            model_prob=order.model_prob,
            market_price=order.market_price,
            reserved_margin=required_margin if order.side == Side.SELL else 0.0,
        )
        
        # Update state
        state.trades.append(trade)
        
        # Update exposure (positive for buys, negative for sells)
        if order.side == Side.BUY:
            state.add_exposure(order.bin_id, order.size)
            state.bankroll -= order.size * fill_price
        else:
            state.add_exposure(order.bin_id, -order.size)
            state.bankroll += order.size * fill_price
            state.reserved_margin += required_margin
    
    def _settle_trades(self, state: SimulatorState, df: pd.DataFrame) -> None:
        """Settle all trades based on actual outcomes."""
        # Build lookup for actual values
        actual_lookup = {}
        for idx in range(len(df)):
            row = df.iloc[idx]
            if "tmax_actual_f" in row and pd.notna(row["tmax_actual_f"]):
                key = (row["station_id"], str(row["target_date_local"]))
                actual_lookup[key] = row["tmax_actual_f"]
        
        # Settle each trade
        for trade in state.trades:
            key = (trade.station_id, str(trade.target_date_local))
            
            if key in actual_lookup:
                actual = actual_lookup[key]
                trade.actual_value = actual
                
                # Determine winning bin
                winning_bin = self.bins.get_winning_bin(actual)
                trade.winning_bin = winning_bin
                
                # Calculate P&L
                if trade.side == "buy":
                    # Bought at fill_price, pays 1 if wins, 0 if loses
                    if trade.bin_id == winning_bin:
                        pnl = trade.size * (1 - trade.fill_price)
                    else:
                        pnl = -trade.size * trade.fill_price
                else:
                    # Sold at fill_price, pays 1 - fill_price if loses, -fill_price if wins
                    if trade.bin_id == winning_bin:
                        pnl = -trade.size * (1 - trade.fill_price)
                    else:
                        pnl = trade.size * trade.fill_price
                
                trade.pnl = pnl
                trade.is_settled = True
                state.bankroll += pnl
                if trade.side == "sell":
                    state.reserved_margin = max(
                        0.0, state.reserved_margin - trade.reserved_margin
                    )
    
    def _build_trades_df(self, state: SimulatorState) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not state.trades:
            return pd.DataFrame(columns=[
                "trade_id", "row_idx", "station_id", "issue_time_utc",
                "target_date_local", "bin_id", "side", "size", "fill_price",
                "edge", "model_prob", "market_price", "actual_value",
                "winning_bin", "pnl", "is_settled", "reserved_margin",
            ])
        
        records = []
        for t in state.trades:
            records.append({
                "trade_id": t.trade_id,
                "row_idx": t.row_idx,
                "station_id": t.station_id,
                "issue_time_utc": t.issue_time_utc,
                "target_date_local": t.target_date_local,
                "bin_id": t.bin_id,
                "side": t.side,
                "size": t.size,
                "fill_price": t.fill_price,
                "edge": t.edge,
                "model_prob": t.model_prob,
                "market_price": t.market_price,
                "actual_value": t.actual_value,
                "winning_bin": t.winning_bin,
                "pnl": t.pnl,
                "is_settled": t.is_settled,
                "reserved_margin": t.reserved_margin,
            })
        
        return pd.DataFrame(records)
    
    def _build_daily_results_df(
        self,
        state: SimulatorState,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build daily P&L summary."""
        if not state.trades:
            return pd.DataFrame(columns=[
                "target_date_local", "station_id", "n_trades", "total_size",
                "total_pnl", "avg_edge", "win_rate",
            ])
        
        # Group trades by date and station
        trades_df = self._build_trades_df(state)
        settled = trades_df[trades_df["is_settled"]].copy()
        
        if settled.empty:
            return pd.DataFrame(columns=[
                "target_date_local", "station_id", "n_trades", "total_size",
                "total_pnl", "avg_edge", "win_rate",
            ])
        
        # Compute daily stats
        daily = settled.groupby(["target_date_local", "station_id"]).agg(
            n_trades=("trade_id", "count"),
            total_size=("size", "sum"),
            total_pnl=("pnl", "sum"),
            avg_edge=("edge", "mean"),
            wins=("pnl", lambda x: (x > 0).sum()),
        ).reset_index()
        
        daily["win_rate"] = daily["wins"] / daily["n_trades"]
        daily = daily.drop(columns=["wins"])
        
        return daily
    
    def _empty_results(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return empty result DataFrames."""
        trades_df = pd.DataFrame(columns=[
            "trade_id", "row_idx", "station_id", "issue_time_utc",
            "target_date_local", "bin_id", "side", "size", "fill_price",
            "edge", "model_prob", "market_price", "actual_value",
            "winning_bin", "pnl", "is_settled", "reserved_margin",
        ])
        daily_df = pd.DataFrame(columns=[
            "target_date_local", "station_id", "n_trades", "total_size",
            "total_pnl", "avg_edge", "win_rate",
        ])
        predictions_df = pd.DataFrame(columns=[
            "row_idx", "station_id", "issue_time_utc", "target_date_local",
            "mu_f", "sigma_f", "tmax_actual_f",
        ])
        return trades_df, daily_df, predictions_df


def run_backtest(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    forecaster: Forecaster,
    uncertainty_model: UncertaintyModel,
    strategy: Strategy,
    price_provider: PriceProvider,
    bins: BinSet,
    slippage: float = 0.01,
    initial_bankroll: float = 10000.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a complete backtest with training and testing.
    
    This is a convenience function that:
    1. Fits the forecaster and uncertainty model on training data
    2. Runs simulation on test data
    
    Args:
        df_train: Training data for fitting models
        df_test: Test data for simulation
        forecaster: Forecaster to fit and use
        uncertainty_model: Uncertainty model to fit and use
        strategy: Trading strategy
        price_provider: Price provider
        bins: Bin definitions
        slippage: Execution slippage
        initial_bankroll: Starting bankroll
        verbose: Whether to print progress
        
    Returns:
        Tuple of (trades_df, daily_results_df, predictions_df)
    """
    # Fit forecaster
    if verbose:
        print("[backtest] Fitting forecaster...")
    forecaster.fit(df_train)
    
    # Compute training residuals for uncertainty model
    train_mu = forecaster.predict_mu(df_train)
    train_residuals = train_mu - df_train["tmax_actual_f"].values
    
    # Fit uncertainty model
    if verbose:
        print("[backtest] Fitting uncertainty model...")
    uncertainty_model.fit(df_train, train_residuals)
    
    # Create and run simulator
    simulator = Simulator(
        forecaster=forecaster,
        uncertainty_model=uncertainty_model,
        strategy=strategy,
        price_provider=price_provider,
        bins=bins,
        slippage=slippage,
        initial_bankroll=initial_bankroll,
    )
    
    return simulator.run(df_test, verbose=verbose)
