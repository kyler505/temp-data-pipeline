"""Bin definitions and probability calculations for Polymarket-style markets.

This module provides:
1. Bin definitions and validation
2. Normal CDF-based probability calculations
3. Utility functions for bin operations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Defer scipy import for optional dependency
_scipy_available = None


def _check_scipy() -> bool:
    """Check if scipy is available."""
    global _scipy_available
    if _scipy_available is None:
        try:
            import scipy
            _scipy_available = True
        except ImportError:
            _scipy_available = False
    return _scipy_available


@dataclass
class Bin:
    """A single temperature bin.
    
    Attributes:
        low: Lower bound (inclusive, -inf for unbounded)
        high: Upper bound (exclusive, inf for unbounded)
        bin_id: Unique identifier for this bin
    """
    low: float
    high: float
    bin_id: int
    
    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"Bin low ({self.low}) must be < high ({self.high})")
    
    def contains(self, value: float) -> bool:
        """Check if value falls in this bin."""
        return self.low <= value < self.high
    
    def label(self) -> str:
        """Human-readable label for this bin."""
        if self.low == float("-inf"):
            return f"< {self.high}°F"
        elif self.high == float("inf"):
            return f">= {self.low}°F"
        else:
            return f"{self.low}-{self.high}°F"


class BinSet:
    """Collection of temperature bins for a market.
    
    Bins should cover the entire temperature range with no gaps or overlaps.
    
    Args:
        bins_f: List of (low, high) tuples defining bin boundaries
    """
    
    def __init__(self, bins_f: Sequence[tuple[float, float]]) -> None:
        if not bins_f:
            raise ValueError("bins_f must not be empty")
        
        # Create Bin objects
        self.bins = [
            Bin(low=lo, high=hi, bin_id=i)
            for i, (lo, hi) in enumerate(bins_f)
        ]
        
        # Validate coverage
        self._validate()
    
    def _validate(self) -> None:
        """Validate that bins cover the range properly."""
        # Sort by low bound
        sorted_bins = sorted(self.bins, key=lambda b: b.low)
        
        # Check for gaps/overlaps
        for i in range(len(sorted_bins) - 1):
            curr = sorted_bins[i]
            next_bin = sorted_bins[i + 1]
            
            if curr.high < next_bin.low:
                raise ValueError(
                    f"Gap between bins: {curr.high} to {next_bin.low}"
                )
            if curr.high > next_bin.low:
                raise ValueError(
                    f"Overlap between bins: {curr.label()} and {next_bin.label()}"
                )
    
    def __len__(self) -> int:
        return len(self.bins)
    
    def __getitem__(self, bin_id: int) -> Bin:
        return self.bins[bin_id]
    
    def __iter__(self):
        return iter(self.bins)
    
    def find_bin(self, value: float) -> Bin | None:
        """Find the bin containing a value.
        
        Args:
            value: Temperature value
            
        Returns:
            Bin containing the value, or None if not found
        """
        for bin_ in self.bins:
            if bin_.contains(value):
                return bin_
        return None
    
    def get_winning_bin(self, actual: float) -> int:
        """Get the bin_id that resolves as winner for an actual value.
        
        Args:
            actual: Actual temperature
            
        Returns:
            bin_id of winning bin
            
        Raises:
            ValueError: If actual doesn't fall in any bin
        """
        bin_ = self.find_bin(actual)
        if bin_ is None:
            raise ValueError(f"Actual value {actual} doesn't fall in any bin")
        return bin_.bin_id
    
    @property
    def labels(self) -> list[str]:
        """Get labels for all bins."""
        return [b.label() for b in self.bins]
    
    @property
    def boundaries(self) -> list[tuple[float, float]]:
        """Get (low, high) boundaries for all bins."""
        return [(b.low, b.high) for b in self.bins]


def compute_bin_probabilities(
    mu: np.ndarray,
    sigma: np.ndarray,
    bins: BinSet | Sequence[tuple[float, float]],
    normalize: bool = True,
) -> np.ndarray:
    """Compute probability of each bin given Normal distribution.
    
    For each prediction (mu, sigma), computes P(bin_i) = P(low_i <= X < high_i)
    where X ~ Normal(mu, sigma).
    
    Args:
        mu: Array of predicted means, shape (n_samples,)
        sigma: Array of predicted std devs, shape (n_samples,)
        bins: BinSet or list of (low, high) tuples
        normalize: If True, renormalize probabilities to sum to 1
        
    Returns:
        Array of probabilities, shape (n_samples, n_bins)
    """
    if not _check_scipy():
        raise ImportError(
            "scipy is required for bin probability computation. "
            "Install with: pip install scipy"
        )
    
    from scipy.stats import norm
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    if mu.shape != sigma.shape:
        raise ValueError(f"mu shape {mu.shape} != sigma shape {sigma.shape}")
    
    # Get bin boundaries
    if isinstance(bins, BinSet):
        boundaries = bins.boundaries
    else:
        boundaries = list(bins)
    
    n_samples = len(mu)
    n_bins = len(boundaries)
    
    probs = np.zeros((n_samples, n_bins))
    
    for i, (lo, hi) in enumerate(boundaries):
        # P(lo <= X < hi) = CDF(hi) - CDF(lo)
        if lo == float("-inf"):
            cdf_lo = np.zeros(n_samples)
        else:
            cdf_lo = norm.cdf(lo, loc=mu, scale=sigma)
        
        if hi == float("inf"):
            cdf_hi = np.ones(n_samples)
        else:
            cdf_hi = norm.cdf(hi, loc=mu, scale=sigma)
        
        probs[:, i] = cdf_hi - cdf_lo
    
    # Normalize if requested
    if normalize:
        row_sums = probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        probs = probs / row_sums
    
    return probs


def compute_brier_score(
    probs: np.ndarray,
    actual: np.ndarray,
    bins: BinSet | Sequence[tuple[float, float]],
) -> float:
    """Compute Brier score for bin probability forecasts.
    
    Brier score = mean((p_predicted - p_actual)^2) across all bins and samples.
    Lower is better (0 is perfect).
    
    Args:
        probs: Predicted probabilities, shape (n_samples, n_bins)
        actual: Actual temperatures, shape (n_samples,)
        bins: BinSet or list of (low, high) tuples
        
    Returns:
        Brier score (lower is better)
    """
    actual = np.asarray(actual)
    
    # Get bin boundaries
    if isinstance(bins, BinSet):
        boundaries = bins.boundaries
        bin_set = bins
    else:
        boundaries = list(bins)
        bin_set = BinSet(boundaries)
    
    n_samples = len(actual)
    n_bins = len(boundaries)
    
    if probs.shape != (n_samples, n_bins):
        raise ValueError(
            f"probs shape {probs.shape} != expected ({n_samples}, {n_bins})"
        )
    
    # Create actual outcome matrix (1 for winning bin, 0 elsewhere)
    actual_probs = np.zeros((n_samples, n_bins))
    for i, val in enumerate(actual):
        if not np.isnan(val):
            winning_bin = bin_set.get_winning_bin(val)
            actual_probs[i, winning_bin] = 1.0
    
    # Remove samples with NaN actual
    valid = ~np.isnan(actual)
    probs_valid = probs[valid]
    actual_probs_valid = actual_probs[valid]
    
    if len(probs_valid) == 0:
        return np.nan
    
    # Brier score = mean squared error between predicted and actual probs
    return np.mean((probs_valid - actual_probs_valid) ** 2)


def compute_log_score(
    probs: np.ndarray,
    actual: np.ndarray,
    bins: BinSet | Sequence[tuple[float, float]],
    eps: float = 1e-10,
) -> float:
    """Compute log score (negative log-likelihood) for bin probability forecasts.
    
    Log score = -mean(log(p_predicted for winning bin)).
    Lower is better (0 would mean perfect probability 1 for correct bin).
    
    Args:
        probs: Predicted probabilities, shape (n_samples, n_bins)
        actual: Actual temperatures, shape (n_samples,)
        bins: BinSet or list of (low, high) tuples
        eps: Small value to avoid log(0)
        
    Returns:
        Log score (lower is better)
    """
    actual = np.asarray(actual)
    
    # Get bin set
    if isinstance(bins, BinSet):
        bin_set = bins
    else:
        bin_set = BinSet(list(bins))
    
    # Compute log score for each sample
    log_scores = []
    for i, val in enumerate(actual):
        if not np.isnan(val):
            winning_bin = bin_set.get_winning_bin(val)
            prob = max(probs[i, winning_bin], eps)
            log_scores.append(-np.log(prob))
    
    if len(log_scores) == 0:
        return np.nan
    
    return np.mean(log_scores)


def create_default_bins(
    temp_range: tuple[float, float] = (20.0, 100.0),
    bin_width: float = 10.0,
) -> list[tuple[float, float]]:
    """Create default temperature bins.
    
    Creates bins covering the temperature range with specified width,
    plus unbounded bins at the extremes.
    
    Args:
        temp_range: (min, max) temperature range for middle bins
        bin_width: Width of each bin in degrees F
        
    Returns:
        List of (low, high) bin boundaries
    """
    lo, hi = temp_range
    boundaries = np.arange(lo, hi + bin_width, bin_width)
    
    bins = []
    # Lower unbounded bin
    bins.append((float("-inf"), boundaries[0]))
    
    # Middle bins
    for i in range(len(boundaries) - 1):
        bins.append((boundaries[i], boundaries[i + 1]))
    
    # Upper unbounded bin
    bins.append((boundaries[-1], float("inf")))
    
    return bins
