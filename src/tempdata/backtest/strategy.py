"""Trading strategies for backtesting.

This module provides:
1. Strategy protocol for trading decision interface
2. Order dataclass for trade instructions
3. EdgeStrategy: Trade when model probability exceeds market price by threshold
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from tempdata.backtest.pricing import MarketPrices


class Side(Enum):
    """Trade side enumeration."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """A trading order for a specific bin.
    
    Attributes:
        bin_id: Which bin to trade
        side: Buy or sell
        size: Number of shares/contracts
        limit_price: Maximum price willing to pay (buy) or minimum to receive (sell)
        edge: Expected edge (model_prob - market_price for buys)
        model_prob: Model's probability for this bin
        market_price: Market price for this bin
    """
    bin_id: int
    side: Side
    size: float
    limit_price: float
    edge: float
    model_prob: float
    market_price: float
    
    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"Order size must be positive, got {self.size}")
        if not 0 <= self.limit_price <= 1:
            raise ValueError(f"Limit price must be in [0, 1], got {self.limit_price}")


@runtime_checkable
class Strategy(Protocol):
    """Protocol for trading strategies.
    
    Strategies decide what trades to make based on model probabilities,
    market prices, and risk constraints.
    """
    
    def generate_orders(
        self,
        row: pd.Series,
        prices: MarketPrices,
        bin_probs: np.ndarray,
        bankroll: float,
        exposure: dict[int, float],
    ) -> list[Order]:
        """Generate trading orders for a single row/market.
        
        Args:
            row: Data row with features (for context)
            prices: Market prices (mid/bid/ask)
            bin_probs: Model's predicted probabilities for each bin
            bankroll: Current available bankroll
            exposure: Current exposure by bin (bin_id -> dollar exposure)
            
        Returns:
            List of Order objects to execute
        """
        ...


class NoOpStrategy:
    """Strategy that never trades. Useful for baseline comparison."""
    
    def generate_orders(
        self,
        row: pd.Series,
        prices: MarketPrices,
        bin_probs: np.ndarray,
        bankroll: float,
        exposure: dict[int, float],
    ) -> list[Order]:
        """Return empty list (no trades)."""
        return []


class EdgeStrategy:
    """Edge-based trading strategy.
    
    Trades when the model's probability exceeds the market price by
    at least edge_min. Uses position limits to manage risk.
    
    Trading logic:
    - BUY when model_prob > market_price + edge_min
    - SELL when model_prob < market_price - edge_min (short selling)
    
    Position sizing:
    - Fixed fraction of bankroll per market
    - Caps on per-bin and total exposure
    
    Args:
        edge_min: Minimum edge required to trade (default 0.03)
        max_per_market_pct: Max bankroll % to risk per market (default 0.02)
        max_total_pct: Max total exposure as % of bankroll (default 0.25)
        allow_shorts: Whether to allow short selling (default False)
        prob_floor: Don't trade bins with prob below this (default 0.02)
        prob_ceiling: Don't trade bins with prob above this (default 0.98)
    """
    
    def __init__(
        self,
        edge_min: float = 0.03,
        max_per_market_pct: float = 0.02,
        max_total_pct: float = 0.25,
        allow_shorts: bool = False,
        prob_floor: float = 0.02,
        prob_ceiling: float = 0.98,
    ) -> None:
        self.edge_min = edge_min
        self.max_per_market_pct = max_per_market_pct
        self.max_total_pct = max_total_pct
        self.allow_shorts = allow_shorts
        self.prob_floor = prob_floor
        self.prob_ceiling = prob_ceiling
    
    def generate_orders(
        self,
        row: pd.Series,
        prices: MarketPrices,
        bin_probs: np.ndarray,
        bankroll: float,
        exposure: dict[int, float],
    ) -> list[Order]:
        """Generate orders based on edge calculation.
        
        Args:
            row: Data row (for context, currently unused)
            prices: Market prices for each bin
            bin_probs: Model's probabilities for each bin
            bankroll: Current bankroll
            exposure: Current exposure by bin
            
        Returns:
            List of orders to execute
        """
        orders = []
        
        # Calculate current total exposure
        total_exposure = sum(abs(e) for e in exposure.values())
        max_total_exposure = bankroll * self.max_total_pct
        
        # Calculate max per-market size
        max_market_size = bankroll * self.max_per_market_pct
        
        # Evaluate each bin
        for bin_id, market_price in prices.mid.items():
            model_prob = bin_probs[bin_id]
            
            # Skip extreme probabilities
            if model_prob < self.prob_floor or model_prob > self.prob_ceiling:
                continue
            
            # Calculate edge
            edge = model_prob - market_price
            
            # Check if edge is sufficient for a buy
            if edge >= self.edge_min:
                # Calculate order size
                remaining_capacity = max_total_exposure - total_exposure
                if remaining_capacity <= 0:
                    continue
                
                size = min(max_market_size, remaining_capacity)
                
                # Adjust for existing exposure in this bin
                current_bin_exposure = exposure.get(bin_id, 0)
                if current_bin_exposure > 0:
                    # Already long, may add less
                    size = min(size, max_market_size - current_bin_exposure)
                
                if size > 0:
                    orders.append(Order(
                        bin_id=bin_id,
                        side=Side.BUY,
                        size=size,
                        limit_price=market_price,  # Will add slippage in simulator
                        edge=edge,
                        model_prob=model_prob,
                        market_price=market_price,
                    ))
                    total_exposure += size
            
            # Check for short opportunity
            elif self.allow_shorts and edge <= -self.edge_min:
                remaining_capacity = max_total_exposure - total_exposure
                if remaining_capacity <= 0:
                    continue
                
                size = min(max_market_size, remaining_capacity)
                
                if size > 0:
                    orders.append(Order(
                        bin_id=bin_id,
                        side=Side.SELL,
                        size=size,
                        limit_price=market_price,
                        edge=-edge,  # Report positive edge
                        model_prob=model_prob,
                        market_price=market_price,
                    ))
                    total_exposure += size
        
        return orders


class KellyStrategy:
    """Kelly criterion-based trading strategy.
    
    Uses fractional Kelly sizing based on edge and odds.
    More sophisticated than fixed sizing but can be aggressive.
    
    Kelly fraction = (p * b - q) / b
    Where:
        p = probability of winning
        b = odds (payout - 1)
        q = 1 - p
    
    Args:
        edge_min: Minimum edge required to trade (default 0.03)
        kelly_fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly)
        max_per_market_pct: Cap on per-market sizing (default 0.05)
        max_total_pct: Cap on total exposure (default 0.25)
    """
    
    def __init__(
        self,
        edge_min: float = 0.03,
        kelly_fraction: float = 0.25,
        max_per_market_pct: float = 0.05,
        max_total_pct: float = 0.25,
    ) -> None:
        self.edge_min = edge_min
        self.kelly_fraction = kelly_fraction
        self.max_per_market_pct = max_per_market_pct
        self.max_total_pct = max_total_pct
    
    def generate_orders(
        self,
        row: pd.Series,
        prices: MarketPrices,
        bin_probs: np.ndarray,
        bankroll: float,
        exposure: dict[int, float],
    ) -> list[Order]:
        """Generate orders using Kelly criterion sizing."""
        orders = []
        
        total_exposure = sum(abs(e) for e in exposure.values())
        max_total_exposure = bankroll * self.max_total_pct
        max_market_size = bankroll * self.max_per_market_pct
        
        for bin_id, market_price in prices.mid.items():
            model_prob = bin_probs[bin_id]
            edge = model_prob - market_price
            
            if edge < self.edge_min:
                continue
            
            # Calculate Kelly fraction
            # For binary outcome: winning pays 1/price - 1, losing pays -1
            # p = model_prob, b = (1/price - 1), q = 1 - model_prob
            if market_price <= 0 or market_price >= 1:
                continue
            
            b = (1 / market_price) - 1  # Odds
            p = model_prob
            q = 1 - p
            
            kelly = (p * b - q) / b if b > 0 else 0
            kelly = max(0, kelly)  # No negative bets
            
            # Apply fractional Kelly
            bet_fraction = kelly * self.kelly_fraction
            
            # Convert to dollar size
            size = bankroll * bet_fraction
            
            # Apply caps
            size = min(size, max_market_size)
            remaining_capacity = max_total_exposure - total_exposure
            size = min(size, remaining_capacity)
            
            if size > 0.01:  # Minimum size threshold
                orders.append(Order(
                    bin_id=bin_id,
                    side=Side.BUY,
                    size=size,
                    limit_price=market_price,
                    edge=edge,
                    model_prob=model_prob,
                    market_price=market_price,
                ))
                total_exposure += size
        
        return orders


def create_strategy(
    strategy_type: str = "edge",
    edge_min: float = 0.03,
    max_per_market_pct: float = 0.02,
    max_total_pct: float = 0.25,
    **kwargs,
) -> Strategy:
    """Factory function to create strategy by type.
    
    Args:
        strategy_type: Strategy type ("edge", "kelly", or "noop")
        edge_min: Minimum edge to trade
        max_per_market_pct: Max size per market
        max_total_pct: Max total exposure
        **kwargs: Additional strategy-specific arguments
        
    Returns:
        Configured Strategy instance
    """
    if strategy_type == "edge":
        return EdgeStrategy(
            edge_min=edge_min,
            max_per_market_pct=max_per_market_pct,
            max_total_pct=max_total_pct,
            **kwargs,
        )
    elif strategy_type == "kelly":
        return KellyStrategy(
            edge_min=edge_min,
            max_per_market_pct=max_per_market_pct,
            max_total_pct=max_total_pct,
            **kwargs,
        )
    elif strategy_type == "noop":
        return NoOpStrategy()
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}")
