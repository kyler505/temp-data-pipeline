"""Forecast models for backtesting.

This module provides:
1. Forecaster protocol for model interface
2. RidgeForecaster: Bias-corrected Ridge regression baseline
3. PassthroughForecaster: Uses raw forecast predictions (no correction)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

# Defer sklearn import for optional dependency
_sklearn_available = None


def _check_sklearn() -> bool:
    """Check if scikit-learn is available."""
    global _sklearn_available
    if _sklearn_available is None:
        try:
            import sklearn
            _sklearn_available = True
        except ImportError:
            _sklearn_available = False
    return _sklearn_available


@runtime_checkable
class Forecaster(Protocol):
    """Protocol for forecast models.
    
    Models produce point predictions (mu) for tmax_actual_f.
    """
    
    def fit(self, df_train: pd.DataFrame) -> None:
        """Fit the model on training data.
        
        Args:
            df_train: Training DataFrame with features and tmax_actual_f label
        """
        ...
    
    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        """Predict mean temperature (mu).
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of predicted temperatures (°F)
        """
        ...


class PassthroughForecaster:
    """Passthrough model that uses raw forecast predictions.
    
    This is the simplest baseline - just use tmax_pred_f directly.
    Useful for benchmarking more complex models.
    """
    
    def fit(self, df_train: pd.DataFrame) -> None:
        """No fitting needed for passthrough."""
        pass
    
    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw forecast predictions."""
        return df["tmax_pred_f"].values.astype(np.float64)


class RidgeForecaster:
    """Bias-corrected Ridge regression forecaster.
    
    Learns to correct systematic forecast biases using:
    - Raw forecast (tmax_pred_f)
    - Seasonal features (sin_doy, cos_doy)
    - Rolling bias features (bias_7d, bias_14d)
    
    Args:
        alpha: Ridge regularization parameter (default 1.0)
        features: List of feature columns to use
        handle_missing: How to handle missing feature values
            - "fill_zero": Fill NaN with 0
            - "fill_mean": Fill NaN with training mean
            - "raise": Raise error on NaN
    """
    
    DEFAULT_FEATURES = ["tmax_pred_f", "sin_doy", "cos_doy", "bias_7d", "bias_14d"]
    
    def __init__(
        self,
        alpha: float = 1.0,
        features: list[str] | None = None,
        handle_missing: str = "fill_zero",
    ) -> None:
        if not _check_sklearn():
            raise ImportError(
                "scikit-learn is required for RidgeForecaster. "
                "Install with: pip install scikit-learn"
            )
        
        self.alpha = alpha
        self.features = features or self.DEFAULT_FEATURES.copy()
        self.handle_missing = handle_missing
        self._model = None
        self._feature_means: dict[str, float] = {}
        self._is_fitted = False
    
    def fit(self, df_train: pd.DataFrame) -> None:
        """Fit Ridge regression on training data.
        
        Args:
            df_train: Training DataFrame with features and tmax_actual_f
        """
        from sklearn.linear_model import Ridge
        
        # Validate required columns
        missing_cols = set(self.features) - set(df_train.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        if "tmax_actual_f" not in df_train.columns:
            raise ValueError("Missing target column: tmax_actual_f")
        
        # Extract features and target
        X = df_train[self.features].copy()
        y = df_train["tmax_actual_f"].values
        
        # Handle missing values
        X = self._handle_missing_train(X)
        
        # Remove rows where target is NaN
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(y) == 0:
            raise ValueError("No valid training samples after removing NaN targets")
        
        # Fit model
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X, y)
        self._is_fitted = True
    
    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        """Predict mean temperature using fitted model.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of predicted temperatures (°F)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Extract features
        X = df[self.features].copy()
        
        # Handle missing values
        X = self._handle_missing_predict(X)
        
        return self._model.predict(X)
    
    def _handle_missing_train(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values during training."""
        if self.handle_missing == "raise":
            if X.isna().any().any():
                nan_cols = X.columns[X.isna().any()].tolist()
                raise ValueError(f"NaN values in feature columns: {nan_cols}")
            return X
        
        elif self.handle_missing == "fill_zero":
            return X.fillna(0)
        
        elif self.handle_missing == "fill_mean":
            # Compute and store means
            for col in X.columns:
                self._feature_means[col] = X[col].mean()
            return X.fillna(self._feature_means)
        
        else:
            raise ValueError(f"Unknown handle_missing: {self.handle_missing}")
    
    def _handle_missing_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values during prediction."""
        if self.handle_missing == "raise":
            if X.isna().any().any():
                nan_cols = X.columns[X.isna().any()].tolist()
                raise ValueError(f"NaN values in feature columns: {nan_cols}")
            return X
        
        elif self.handle_missing == "fill_zero":
            return X.fillna(0)
        
        elif self.handle_missing == "fill_mean":
            return X.fillna(self._feature_means)
        
        else:
            raise ValueError(f"Unknown handle_missing: {self.handle_missing}")
    
    @property
    def coef_(self) -> np.ndarray | None:
        """Get model coefficients (if fitted)."""
        if self._model is None:
            return None
        return self._model.coef_
    
    @property
    def intercept_(self) -> float | None:
        """Get model intercept (if fitted)."""
        if self._model is None:
            return None
        return self._model.intercept_
    
    def feature_importance(self) -> pd.Series | None:
        """Get feature importances as coefficients.
        
        Returns:
            Series mapping feature names to coefficients, or None if not fitted
        """
        if not self._is_fitted:
            return None
        return pd.Series(self.coef_, index=self.features, name="coefficient")


def create_forecaster(
    model_type: str,
    alpha: float = 1.0,
    features: list[str] | None = None,
) -> Forecaster:
    """Factory function to create forecaster by type.
    
    Args:
        model_type: Model type identifier ("ridge" or "passthrough")
        alpha: Ridge regularization parameter
        features: Feature columns for Ridge model
        
    Returns:
        Configured Forecaster instance
    """
    if model_type == "ridge":
        return RidgeForecaster(alpha=alpha, features=features)
    elif model_type == "passthrough":
        return PassthroughForecaster()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
