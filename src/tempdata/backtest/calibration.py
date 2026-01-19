"""Uncertainty calibration models for backtesting.

This module provides:
1. UncertaintyModel protocol for sigma estimation
2. BucketedSigma: Residual std dev by lead_hours bucket
3. GlobalSigma: Single sigma for all predictions
4. RollingSigma: Uses pre-computed sigma_lead from features
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class UncertaintyModel(Protocol):
    """Protocol for uncertainty/sigma estimation models.
    
    Models produce standard deviation estimates for each prediction,
    enabling probabilistic forecasts and bin probability calculation.
    """
    
    def fit(self, df_train: pd.DataFrame, residuals: np.ndarray) -> None:
        """Fit the uncertainty model on training residuals.
        
        Args:
            df_train: Training DataFrame with features
            residuals: Array of (predicted - actual) residuals
        """
        ...
    
    def predict_sigma(self, df: pd.DataFrame) -> np.ndarray:
        """Predict standard deviation (sigma) for each row.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of sigma values (°F)
        """
        ...


class GlobalSigma:
    """Simple global sigma model.
    
    Uses a single sigma value computed from training residuals.
    This is the simplest baseline for uncertainty estimation.
    
    Args:
        sigma_floor: Minimum sigma to prevent overconfidence (default 1.0)
    """
    
    def __init__(self, sigma_floor: float = 1.0) -> None:
        self.sigma_floor = sigma_floor
        self._sigma: float | None = None
        self._is_fitted = False
    
    def fit(self, df_train: pd.DataFrame, residuals: np.ndarray) -> None:
        """Compute global sigma from training residuals."""
        valid_residuals = residuals[~np.isnan(residuals)]
        if len(valid_residuals) < 2:
            raise ValueError("Need at least 2 valid residuals to estimate sigma")
        
        self._sigma = max(np.std(valid_residuals, ddof=1), self.sigma_floor)
        self._is_fitted = True
    
    def predict_sigma(self, df: pd.DataFrame) -> np.ndarray:
        """Return global sigma for all rows."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return np.full(len(df), self._sigma)
    
    @property
    def sigma(self) -> float | None:
        """Get the fitted sigma value."""
        return self._sigma


class BucketedSigma:
    """Sigma model with lead_hours bucketing.
    
    Computes separate sigma for different lead_hours ranges,
    capturing the typical pattern of uncertainty increasing with lead time.
    
    Args:
        buckets: List of (min_lead, max_lead) tuples defining buckets.
            If None, uses default buckets.
        sigma_floor: Minimum sigma to prevent overconfidence (default 1.0)
    """
    
    DEFAULT_BUCKETS = [
        (0, 24),
        (24, 48),
        (48, 72),
        (72, float("inf")),
    ]
    
    def __init__(
        self,
        buckets: list[tuple[int, int]] | None = None,
        sigma_floor: float = 1.0,
    ) -> None:
        self.buckets = buckets or self.DEFAULT_BUCKETS.copy()
        self.sigma_floor = sigma_floor
        self._bucket_sigmas: dict[int, float] = {}  # bucket_idx -> sigma
        self._global_sigma: float | None = None  # fallback
        self._is_fitted = False
    
    def fit(self, df_train: pd.DataFrame, residuals: np.ndarray) -> None:
        """Compute sigma for each lead_hours bucket.
        
        Args:
            df_train: Training DataFrame with lead_hours column
            residuals: Array of (predicted - actual) residuals
        """
        if "lead_hours" not in df_train.columns:
            raise ValueError("DataFrame must have lead_hours column")
        
        if len(df_train) != len(residuals):
            raise ValueError("df_train and residuals must have same length")
        
        lead_hours = df_train["lead_hours"].values
        
        # Compute sigma for each bucket
        for bucket_idx, (lo, hi) in enumerate(self.buckets):
            mask = (lead_hours >= lo) & (lead_hours < hi) & ~np.isnan(residuals)
            bucket_residuals = residuals[mask]
            
            if len(bucket_residuals) >= 2:
                sigma = np.std(bucket_residuals, ddof=1)
                self._bucket_sigmas[bucket_idx] = max(sigma, self.sigma_floor)
            # If not enough data, will fall back to global sigma
        
        # Compute global sigma as fallback
        valid_residuals = residuals[~np.isnan(residuals)]
        if len(valid_residuals) >= 2:
            self._global_sigma = max(np.std(valid_residuals, ddof=1), self.sigma_floor)
        else:
            self._global_sigma = self.sigma_floor
        
        # Fill missing buckets with global sigma
        for bucket_idx in range(len(self.buckets)):
            if bucket_idx not in self._bucket_sigmas:
                self._bucket_sigmas[bucket_idx] = self._global_sigma
        
        self._is_fitted = True
    
    def predict_sigma(self, df: pd.DataFrame) -> np.ndarray:
        """Predict sigma based on lead_hours bucket.
        
        Args:
            df: DataFrame with lead_hours column
            
        Returns:
            Array of sigma values (°F)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if "lead_hours" not in df.columns:
            raise ValueError("DataFrame must have lead_hours column")
        
        lead_hours = df["lead_hours"].values
        sigmas = np.full(len(df), self._global_sigma)
        
        for bucket_idx, (lo, hi) in enumerate(self.buckets):
            mask = (lead_hours >= lo) & (lead_hours < hi)
            sigmas[mask] = self._bucket_sigmas[bucket_idx]
        
        return sigmas
    
    def get_bucket_sigmas(self) -> dict[tuple[int, int], float]:
        """Get sigma values by bucket range.
        
        Returns:
            Dictionary mapping (lo, hi) bucket to sigma
        """
        if not self._is_fitted:
            return {}
        return {
            self.buckets[idx]: sigma
            for idx, sigma in self._bucket_sigmas.items()
        }


class RollingSigma:
    """Sigma model using pre-computed sigma_lead from features.
    
    Uses the sigma_lead column from the feature engineering pipeline,
    which contains expanding window std dev by lead_hours.
    
    Args:
        sigma_floor: Minimum sigma to prevent overconfidence (default 1.0)
        fallback_sigma: Sigma to use when sigma_lead is NaN (default 3.0)
    """
    
    def __init__(
        self,
        sigma_floor: float = 1.0,
        fallback_sigma: float = 3.0,
    ) -> None:
        self.sigma_floor = sigma_floor
        self.fallback_sigma = fallback_sigma
        self._is_fitted = False
    
    def fit(self, df_train: pd.DataFrame, residuals: np.ndarray) -> None:
        """No fitting needed - uses pre-computed sigma_lead."""
        # Verify sigma_lead column exists
        if "sigma_lead" not in df_train.columns:
            raise ValueError("DataFrame must have sigma_lead column")
        self._is_fitted = True
    
    def predict_sigma(self, df: pd.DataFrame) -> np.ndarray:
        """Return sigma_lead values, with floor applied.
        
        Args:
            df: DataFrame with sigma_lead column
            
        Returns:
            Array of sigma values (°F)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if "sigma_lead" not in df.columns:
            raise ValueError("DataFrame must have sigma_lead column")
        
        sigmas = df["sigma_lead"].values.astype(np.float64)
        
        # Replace NaN with fallback
        sigmas = np.where(np.isnan(sigmas), self.fallback_sigma, sigmas)
        
        # Apply floor
        sigmas = np.maximum(sigmas, self.sigma_floor)
        
        return sigmas


def create_uncertainty_model(
    sigma_type: str,
    sigma_buckets: list[tuple[int, int]] | None = None,
    sigma_floor: float = 1.0,
) -> UncertaintyModel:
    """Factory function to create uncertainty model by type.
    
    Args:
        sigma_type: Model type ("global", "bucketed", or "rolling")
        sigma_buckets: Buckets for BucketedSigma
        sigma_floor: Minimum sigma value
        
    Returns:
        Configured UncertaintyModel instance
    """
    if sigma_type == "global":
        return GlobalSigma(sigma_floor=sigma_floor)
    elif sigma_type == "bucketed":
        return BucketedSigma(buckets=sigma_buckets, sigma_floor=sigma_floor)
    elif sigma_type == "rolling":
        return RollingSigma(sigma_floor=sigma_floor)
    else:
        raise ValueError(f"Unknown sigma_type: {sigma_type}")


def compute_interval_coverage(
    actual: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    levels: list[float] | None = None,
) -> dict[float, float]:
    """Compute prediction interval coverage.
    
    Check what fraction of actuals fall within predicted intervals.
    Well-calibrated models should have coverage close to the nominal level.
    
    Args:
        actual: Array of actual values
        mu: Array of predicted means
        sigma: Array of predicted standard deviations
        levels: Coverage levels to check (default [0.5, 0.8, 0.9])
        
    Returns:
        Dictionary mapping level to observed coverage fraction
    """
    if levels is None:
        levels = [0.50, 0.80, 0.90]
    
    # Remove NaN values
    valid = ~(np.isnan(actual) | np.isnan(mu) | np.isnan(sigma))
    actual = actual[valid]
    mu = mu[valid]
    sigma = sigma[valid]
    
    if len(actual) == 0:
        return {level: np.nan for level in levels}
    
    # Import scipy for quantile function
    try:
        from scipy.stats import norm
    except ImportError:
        raise ImportError("scipy is required for interval coverage computation")
    
    coverage = {}
    for level in levels:
        # Compute z-score for this coverage level
        # For 90% coverage, we want the 5th and 95th percentiles
        alpha = (1 - level) / 2
        z = norm.ppf(1 - alpha)
        
        # Check how many actuals fall within [mu - z*sigma, mu + z*sigma]
        lower = mu - z * sigma
        upper = mu + z * sigma
        
        in_interval = (actual >= lower) & (actual <= upper)
        coverage[level] = in_interval.mean()
    
    return coverage
