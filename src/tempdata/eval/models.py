"""Forecast models for temperature prediction.

This module provides forecasting models that produce point predictions (mu).
No trading concepts - pure temperature prediction only.

Models:
- PassthroughForecaster: Uses raw forecast as prediction (baseline)
- RidgeForecaster: Ridge regression bias correction
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Forecaster(Protocol):
    """Protocol for forecasting models.

    Models produce point predictions (mu) for temperature.
    """

    def fit(self, df_train: pd.DataFrame) -> None:
        """Fit the model on training data.

        Args:
            df_train: Training DataFrame with features and tmax_actual_f
        """
        ...

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        """Predict point estimates (mu) for temperature.

        Args:
            df: DataFrame with features

        Returns:
            Array of predicted temperatures (°F)
        """
        ...


class PassthroughForecaster:
    """Passthrough forecaster that returns the raw forecast.

    This is the simplest baseline - just returns tmax_pred_f as-is.
    Useful for comparing with corrected models.
    """

    def __init__(self, pred_col: str = "tmax_pred_f") -> None:
        """Initialize passthrough forecaster.

        Args:
            pred_col: Column name for predictions (default: tmax_pred_f)
        """
        self.pred_col = pred_col

    def fit(self, df_train: pd.DataFrame) -> None:
        """No fitting needed for passthrough."""
        pass

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw forecast as prediction.

        Args:
            df: DataFrame with pred_col

        Returns:
            Array of raw forecast values
        """
        return df[self.pred_col].values


class RidgeForecaster:
    """Ridge regression forecaster for bias correction.

    Fits a ridge regression model to predict actual temperature
    from forecast and other features. This provides bias correction
    and can incorporate seasonal patterns.

    Default features:
    - tmax_pred_f: Raw forecast
    - sin_doy, cos_doy: Seasonal harmonics
    - bias_7d, bias_14d: Rolling bias features
    """

    DEFAULT_FEATURES = ["tmax_pred_f", "sin_doy", "cos_doy", "bias_7d", "bias_14d"]

    def __init__(
        self,
        alpha: float = 1.0,
        features: list[str] | None = None,
        handle_missing: str = "fill_zero",
    ) -> None:
        """Initialize ridge forecaster.

        Args:
            alpha: Ridge regularization parameter
            features: Feature columns to use (default: DEFAULT_FEATURES)
            handle_missing: How to handle missing values ("fill_zero" or "drop")
        """
        self.alpha = alpha
        self.features = features or self.DEFAULT_FEATURES.copy()
        self.handle_missing = handle_missing
        self.model = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, df_train: pd.DataFrame) -> None:
        """Fit ridge regression on training data.

        Args:
            df_train: Training DataFrame with features and tmax_actual_f
        """
        from sklearn.linear_model import Ridge

        X = self._prepare_features(df_train)
        y = df_train["tmax_actual_f"].values

        # Handle missing values
        if self.handle_missing == "fill_zero":
            X = np.nan_to_num(X, nan=0.0)
        elif self.handle_missing == "drop":
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[mask]
            y = y[mask]

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)

        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        """Predict temperature using fitted model.

        Args:
            df: DataFrame with features

        Returns:
            Array of predicted temperatures (°F)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._prepare_features(df)

        if self.handle_missing == "fill_zero":
            X = np.nan_to_num(X, nan=0.0)

        return self.model.predict(X)

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Feature matrix (n_samples, n_features)
        """
        available = [f for f in self.features if f in df.columns]
        if not available:
            raise ValueError(
                f"No features found. Expected: {self.features}, "
                f"Got: {list(df.columns)}"
            )
        return df[available].values


def create_forecaster(
    model_type: str,
    alpha: float = 1.0,
    features: list[str] | None = None,
) -> Forecaster:
    """Factory function to create forecaster by type.

    Args:
        model_type: Model type ("passthrough" or "ridge")
        alpha: Ridge regularization parameter
        features: Feature columns for ridge model

    Returns:
        Configured Forecaster instance
    """
    if model_type == "passthrough":
        return PassthroughForecaster()
    elif model_type == "ridge":
        return RidgeForecaster(alpha=alpha, features=features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
