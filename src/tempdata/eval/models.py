"""Forecast models for temperature prediction.

This module provides forecasting models that produce point predictions (mu).
No trading concepts - pure temperature prediction only.

Models:
- PassthroughForecaster: Uses raw forecast as prediction (baseline)
- RidgeForecaster: Ridge regression bias correction
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Literal, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb


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


class XGBoostForecaster:
    """Forecasting model using XGBoost.

    Uses gradient boosted trees to model non-linear relationships.
    """

    def __init__(
        self,
        features: list[str] | None = None,
        hyperparams: dict[str, Any] | None = None
    ) -> None:
        """Initialize forecaster.

        Args:
            features: List of feature columns to use. Defaults to standard set.
            hyperparams: Dictionary of XGBoost hyperparameters.
        """
        self.features = features or [
            "tmax_pred_f", "sin_doy", "cos_doy", "bias_7d", "bias_14d", "lead_hours"
        ]
        self.params = {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "early_stopping_rounds": 10
        }
        if hyperparams:
            self.params.update(hyperparams)

        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, df_train: pd.DataFrame) -> None:
        """Fit the model on training data.

        Args:
            df_train: DataFrame with features and tmax_actual_f
        """
        # Ensure all features exist
        available_features = [f for f in self.features if f in df_train.columns]
        if not available_features:
            raise ValueError(f"None of the requested features {self.features} are in the dataframe used for training.")

        X = df_train[available_features]
        y = df_train["tmax_actual_f"]

        # Simple fit without validation sets for now, as validation is handled by the runner spliter
        # In a more advanced setup, we could pass val set here for early_stopping
        self.model.fit(X, y)

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        """Predict point estimates (mu) for temperature.

        Args:
            df: DataFrame with features

        Returns:
            Array of predicted temperatures
        """
        available_features = [f for f in self.features if f in df.columns]
        if not available_features:
             raise ValueError(f"None of the requested features {self.features} are in the dataframe used for prediction.")

        X = df[available_features]
        return self.model.predict(X)


def create_forecaster(
    model_type: str,
    alpha: float = 1.0,
    features: list[str] | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> Forecaster:
    """Factory function to create forecaster by type.

    Args:
        model_type: Model type ("passthrough", "ridge", "persistence", "knn", "xgboost")
        alpha: Ridge regularization parameter
        features: Feature columns for model
        hyperparams: Dictionary of hyperparameters for models that support it

    Returns:
        Configured Forecaster instance
    """
    if model_type == "passthrough":
        return PassthroughForecaster()
    elif model_type == "ridge":
        return RidgeForecaster(alpha=alpha, features=features)
    elif model_type == "persistence":
        return PersistenceForecaster()
    elif model_type == "knn":
        return KNNForecaster(features=features)
    elif model_type == "xgboost":
        return XGBoostForecaster(features=features, hyperparams=hyperparams)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class PersistenceForecaster:
    """Persistence forecaster.

    Predicts today's Tmax using yesterday's observed Tmax (or similar simple logic).

    For this implementation, we assume 'persistence' means:
    Predicted Tmax = Tmax of the previous day (relative to target_date).

    Requires features: ['tmax_actual_f_lag1'] or similar.
    However, standard persistence usually means 'last available observation'.

    If 'tmax_actual_f_lag1' is available in features, we use it.
    Otherwise, we might need to rely on the pipeline providing it.
    """

    def fit(self, df_train: pd.DataFrame) -> None:
        """No fitting needed."""
        pass

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using accumulated persistence (last observed value).

        We expect a feature representing the last observed temperature.
        Let's look for 'tmax_actual_lag_1d' or similar in the dataframe.
        """
        # For now, let's assume valid lag features are passed in df.
        cols = [c for c in df.columns if "lag" in c and "tmax" in c]
        if not cols:
             # Fallback if no specific lag column: use raw forecast as fallback
             # (essentially degrading to passthrough if we can't persist)
             # But strictly, persistence should fail if no history.
             # Let's try to assume 'tmax_actual_f_lag1' exists.
             col = "tmax_actual_f_lag1"
        else:
             # Pick the most recent one (smallest lag)
             col = sorted(cols)[0]

        if col not in df.columns:
             raise ValueError(f"Persistence requires a lag feature (e.g. {col}). Available: {df.columns.tolist()}")

        # Fill missing lags with 0 or drop? Persistence on missing data is undefined.
        # We fill with 0 to match other models' behavior, but this is dangerous.
        return df[col].fillna(0.0).values


class KNNForecaster:
    """K-Nearest Neighbors forecaster.

    Finds similar days in history and aggregates their outcomes.
    """

    DEFAULT_FEATURES = ["tmax_pred_f", "lead_hours", "sin_doy", "cos_doy"]

    def __init__(self, n_neighbors: int = 50, features: list[str] | None = None) -> None:
        self.n_neighbors = n_neighbors
        self.features = features or self.DEFAULT_FEATURES.copy()
        self.model = None

    def fit(self, df_train: pd.DataFrame) -> None:
        from sklearn.neighbors import KNeighborsRegressor

        X = self._prepare_features(df_train)
        y = df_train["tmax_actual_f"].values

        X = np.nan_to_num(X, nan=0.0)

        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)

        # Cap n_neighbors if train size is small
        if len(X) < self.n_neighbors:
            self.model.set_params(n_neighbors=len(X))

        self.model.fit(X, y)

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")

        X = self._prepare_features(df)
        X = np.nan_to_num(X, nan=0.0)
        return self.model.predict(X)

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        # Same helper as Ridge
        available = [f for f in self.features if f in df.columns]
        if not available:
            raise ValueError(f"No features found. Expected: {self.features}")
        return df[available].values
