"""Uncertainty calibration models for temperature evaluation.

This module provides sigma/uncertainty estimation methods:
- GlobalSigma: Single sigma from training residuals
- BucketedSigma: Sigma by lead_hours bucket
- RollingSigma: Uses pre-computed sigma_lead from features

Adapted from backtest/calibration.py with no trading concepts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class UncertaintyModel(Protocol):
    """Protocol for uncertainty/sigma estimation models.

    Models produce standard deviation estimates for each prediction,
    enabling calibration assessment and interval construction.
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
    """

    def __init__(self, sigma_floor: float = 1.0) -> None:
        """Initialize global sigma model.

        Args:
            sigma_floor: Minimum sigma value to prevent overconfidence
        """
        self.sigma_floor = sigma_floor
        self._sigma: float | None = None

    def fit(self, df_train: pd.DataFrame, residuals: np.ndarray) -> None:
        """Compute global sigma from training residuals.

        Args:
            df_train: Training DataFrame (unused)
            residuals: Array of residuals
        """
        raw_sigma = float(np.std(residuals))
        self._sigma = max(raw_sigma, self.sigma_floor)

    def predict_sigma(self, df: pd.DataFrame) -> np.ndarray:
        """Return global sigma for all rows.

        Args:
            df: DataFrame (only length matters)

        Returns:
            Array of identical sigma values
        """
        if self._sigma is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return np.full(len(df), self._sigma)

    @property
    def sigma(self) -> float:
        """Get the fitted sigma value."""
        if self._sigma is None:
            raise RuntimeError("Model not fitted.")
        return self._sigma


class BucketedSigma:
    """Sigma model with lead_hours bucketing.

    Computes separate sigma for different lead_hours ranges,
    capturing the typical pattern of uncertainty increasing with lead time.
    """

    DEFAULT_BUCKETS = [(0, 36), (36, 60), (60, 84), (84, 120)]

    def __init__(
        self,
        buckets: list[tuple[int, int]] | None = None,
        sigma_floor: float = 1.0,
    ) -> None:
        """Initialize bucketed sigma model.

        Args:
            buckets: List of (low, high) lead hour buckets
            sigma_floor: Minimum sigma value
        """
        self.buckets = buckets or self.DEFAULT_BUCKETS.copy()
        self.sigma_floor = sigma_floor
        self._bucket_sigmas: dict[tuple[int, int], float] = {}
        self._fallback_sigma: float | None = None

    def fit(self, df_train: pd.DataFrame, residuals: np.ndarray) -> None:
        """Compute sigma for each lead_hours bucket.

        Args:
            df_train: Training DataFrame with lead_hours column
            residuals: Array of (predicted - actual) residuals
        """
        if "lead_hours" not in df_train.columns:
            # Fall back to global sigma
            self._fallback_sigma = max(float(np.std(residuals)), self.sigma_floor)
            return

        lead_hours = df_train["lead_hours"].values

        # Compute sigma for each bucket
        for lo, hi in self.buckets:
            mask = (lead_hours >= lo) & (lead_hours < hi)
            if mask.sum() >= 10:  # Minimum samples for stable estimate
                bucket_std = float(np.std(residuals[mask]))
                self._bucket_sigmas[(lo, hi)] = max(bucket_std, self.sigma_floor)

        # Fallback for any bucket without enough data
        self._fallback_sigma = max(float(np.std(residuals)), self.sigma_floor)

    def predict_sigma(self, df: pd.DataFrame) -> np.ndarray:
        """Predict sigma based on lead_hours bucket.

        Args:
            df: DataFrame with lead_hours column

        Returns:
            Array of sigma values (°F)
        """
        if self._fallback_sigma is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if "lead_hours" not in df.columns or not self._bucket_sigmas:
            return np.full(len(df), self._fallback_sigma)

        lead_hours = df["lead_hours"].values
        sigmas = np.full(len(df), self._fallback_sigma)

        for (lo, hi), sigma in self._bucket_sigmas.items():
            mask = (lead_hours >= lo) & (lead_hours < hi)
            sigmas[mask] = sigma

        return sigmas

    def get_bucket_sigmas(self) -> dict[tuple[int, int], float]:
        """Get sigma values by bucket range.

        Returns:
            Dictionary mapping (lo, hi) bucket to sigma
        """
        return self._bucket_sigmas.copy()


class RollingSigma:
    """Sigma model using pre-computed sigma_lead from features.

    Uses the sigma_lead column from the feature engineering pipeline,
    which contains expanding window std dev by lead_hours.
    """

    def __init__(
        self,
        sigma_floor: float = 1.0,
        fallback_sigma: float = 3.0,
    ) -> None:
        """Initialize rolling sigma model.

        Args:
            sigma_floor: Minimum sigma value
            fallback_sigma: Sigma to use if sigma_lead is missing/NaN
        """
        self.sigma_floor = sigma_floor
        self.fallback_sigma = fallback_sigma

    def fit(self, df_train: pd.DataFrame, residuals: np.ndarray) -> None:
        """No fitting needed - uses pre-computed sigma_lead."""
        pass

    def predict_sigma(self, df: pd.DataFrame) -> np.ndarray:
        """Return sigma_lead values, with floor applied.

        Args:
            df: DataFrame with sigma_lead column

        Returns:
            Array of sigma values (°F)
        """
        if "sigma_lead" not in df.columns:
            return np.full(len(df), self.fallback_sigma)

        sigmas = df["sigma_lead"].values.copy()
        sigmas = np.where(np.isnan(sigmas), self.fallback_sigma, sigmas)
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
        raise ValueError(f"Unknown sigma type: {sigma_type}")
