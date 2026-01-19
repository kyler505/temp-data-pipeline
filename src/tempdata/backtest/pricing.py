"""Market pricing models for backtesting.

This module provides:
1. PriceProvider protocol for market price generation
2. SyntheticPriceProvider: Generates realistic synthetic prices
3. FixedSpreadPriceProvider: Uses model probabilities with fixed spread
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketPrices:
    """Bid/ask/mid prices for each bin."""
    mid: dict[int, float]
    bid: dict[int, float]
    ask: dict[int, float]


@runtime_checkable
class PriceProvider(Protocol):
    """Protocol for market price providers.
    
    Price providers generate market prices for each bin, which are used
    to compute expected value and make trading decisions.
    """
    
    def get_prices(
        self,
        row: pd.Series,
        bin_probs: np.ndarray,
    ) -> MarketPrices:
        """Get market prices for each bin.
        
        Args:
            row: Data row with features (for context)
            bin_probs: Model's predicted probabilities for each bin
            
        Returns:
            MarketPrices with mid/bid/ask quotes (0-1)
        """
        ...


class SyntheticPriceProvider:
    """Synthetic market price generator.
    
    Generates realistic market prices by adding noise and spread to
    the model's predicted probabilities. This simulates a market that
    is informed but not identical to our model.
    
    Price formula:
        price = baseline_prob + noise + spread_adjustment
    
    Where:
        - baseline_prob is either the model prob or a perturbed version
        - noise is random (simulates market uncertainty)
        - spread_adjustment accounts for bid/ask spread
    
    Args:
        noise_std: Standard deviation of price noise (default 0.05)
        spread: Bid-ask spread to apply (default 0.02)
        market_skill: How much market uses true prob vs model prob.
            0 = fully efficient (uses true baseline), 1 = follows model
        random_seed: Seed for reproducibility
    """
    
    def __init__(
        self,
        noise_std: float = 0.05,
        spread: float = 0.02,
        market_skill: float = 0.5,
        random_seed: int = 42,
    ) -> None:
        self.noise_std = noise_std
        self.spread = spread
        self.market_skill = market_skill
        self._rng = np.random.default_rng(random_seed)
    
    def get_prices(
        self,
        row: pd.Series,
        bin_probs: np.ndarray,
    ) -> MarketPrices:
        """Generate synthetic market prices.
        
        Args:
            row: Data row (used for seed variation)
            bin_probs: Model's predicted probabilities
            
        Returns:
            Dictionary mapping bin_id to price
        """
        n_bins = len(bin_probs)

        # Start with model probabilities as baseline
        baseline = bin_probs.copy()

        # Add noise
        noise = self._rng.normal(0, self.noise_std, n_bins)
        mid = baseline + noise

        # Clip to valid range
        mid = np.clip(mid, 0.01, 0.99)

        # Normalize mid prices to sum to 1
        mid = mid / mid.sum()

        bid, ask = _apply_spread(mid, self.spread)

        return MarketPrices(
            mid=_to_price_dict(mid),
            bid=_to_price_dict(bid),
            ask=_to_price_dict(ask),
        )
    
    def set_seed(self, seed: int) -> None:
        """Reset the random number generator with a new seed."""
        self._rng = np.random.default_rng(seed)


class FixedSpreadPriceProvider:
    """Simple price provider using model probabilities with fixed spread.
    
    Prices are set to model probabilities, representing a market that
    perfectly agrees with our model. The spread is applied during
    execution (not here).
    
    This is useful for testing strategy logic without market uncertainty.
    """
    
    def __init__(self, spread: float = 0.0) -> None:
        self.spread = spread

    def get_prices(
        self,
        row: pd.Series,
        bin_probs: np.ndarray,
    ) -> MarketPrices:
        """Return model probabilities as prices.
        
        Args:
            row: Data row (unused)
            bin_probs: Model's predicted probabilities
            
        Returns:
            Dictionary mapping bin_id to probability/price
        """
        mid = np.clip(bin_probs, 0.01, 0.99)
        mid = mid / mid.sum()
        bid, ask = _apply_spread(mid, self.spread)

        return MarketPrices(
            mid=_to_price_dict(mid),
            bid=_to_price_dict(bid),
            ask=_to_price_dict(ask),
        )


class AdversarialPriceProvider:
    """Price provider that is systematically smarter than our model.
    
    This simulates a "Hard Mode" where the market has better information.
    Useful for stress testing strategies.
    
    The market price moves toward the true outcome, making it harder
    for our model to find edge.
    
    Args:
        noise_std: Standard deviation of price noise
        market_edge: How much market knows truth (0 = same as us, 1 = perfect)
        random_seed: Seed for reproducibility
    """
    
    def __init__(
        self,
        noise_std: float = 0.03,
        market_edge: float = 0.2,
        spread: float = 0.0,
        random_seed: int = 42,
    ) -> None:
        self.noise_std = noise_std
        self.market_edge = market_edge
        self.spread = spread
        self._rng = np.random.default_rng(random_seed)
        self._true_outcomes: dict[str, int] = {}  # Cached true outcomes
    
    def set_true_outcome(self, row_key: str, winning_bin: int) -> None:
        """Set the true outcome for a row (for adversarial pricing).
        
        Args:
            row_key: Unique identifier for the row
            winning_bin: The bin that actually wins
        """
        self._true_outcomes[row_key] = winning_bin
    
    def get_prices(
        self,
        row: pd.Series,
        bin_probs: np.ndarray,
    ) -> MarketPrices:
        """Generate adversarial market prices.
        
        If true outcome is known, market price moves toward it.
        
        Args:
            row: Data row (needs unique identifier for outcome lookup)
            bin_probs: Model's predicted probabilities
            
        Returns:
            Dictionary mapping bin_id to price
        """
        n_bins = len(bin_probs)
        
        # Create row key for outcome lookup
        row_key = self._get_row_key(row)
        
        # Start with model probabilities
        prices = bin_probs.copy()
        
        # If we know the true outcome, move prices toward it
        if row_key in self._true_outcomes:
            winning_bin = self._true_outcomes[row_key]
            true_probs = np.zeros(n_bins)
            true_probs[winning_bin] = 1.0
            
            # Blend model probs with truth
            prices = (1 - self.market_edge) * prices + self.market_edge * true_probs
        
        # Add small noise
        noise = self._rng.normal(0, self.noise_std, n_bins)
        prices = prices + noise
        
        # Clip and normalize mid prices
        mid = np.clip(prices, 0.01, 0.99)
        mid = mid / mid.sum()

        bid, ask = _apply_spread(mid, self.spread)

        return MarketPrices(
            mid=_to_price_dict(mid),
            bid=_to_price_dict(bid),
            ask=_to_price_dict(ask),
        )
    
    def _get_row_key(self, row: pd.Series) -> str:
        """Generate unique key for a row."""
        # Use station_id + issue_time + target_date as key
        parts = []
        if "station_id" in row:
            parts.append(str(row["station_id"]))
        if "issue_time_utc" in row:
            parts.append(str(row["issue_time_utc"]))
        if "target_date_local" in row:
            parts.append(str(row["target_date_local"]))
        return "_".join(parts) if parts else str(row.name)


def create_price_provider(
    price_type: str,
    noise: float = 0.05,
    spread: float = 0.02,
    random_seed: int = 42,
) -> PriceProvider:
    """Factory function to create price provider by type.
    
    Args:
        price_type: Provider type ("synthetic", "fixed", or "adversarial")
        noise: Noise standard deviation for synthetic/adversarial
        spread: Spread for synthetic provider
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured PriceProvider instance
    """
    if price_type == "synthetic":
        return SyntheticPriceProvider(
            noise_std=noise,
            spread=spread,
            random_seed=random_seed,
        )
    elif price_type == "fixed":
        return FixedSpreadPriceProvider(spread=spread)
    elif price_type == "adversarial":
        return AdversarialPriceProvider(
            noise_std=noise,
            spread=spread,
            random_seed=random_seed,
        )
    else:
        raise ValueError(f"Unknown price_type: {price_type}")


def _apply_spread(mid: np.ndarray, spread: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply a symmetric bid/ask spread around mid prices."""
    half_spread = max(spread, 0.0) / 2
    bid = np.clip(mid - half_spread, 0.01, 0.99)
    ask = np.clip(mid + half_spread, 0.01, 0.99)
    return bid, ask


def _to_price_dict(values: np.ndarray) -> dict[int, float]:
    return {i: float(p) for i, p in enumerate(values)}
