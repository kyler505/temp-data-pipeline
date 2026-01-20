"""Tests for temperature evaluation module.

Key tests:
- Config validation
- Data loading and filtering
- Time split correctness (no leakage)
- Model fitting and prediction
- Metric computation
- Uncertainty calibration
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest


class TestEvalConfig:
    """Tests for EvalConfig dataclass."""

    def test_valid_config(self) -> None:
        """Valid config should pass validation."""
        from tempdata.eval.config import EvalConfig

        config = EvalConfig(
            run_name="test_run",
            station_ids=["KLGA"],
            start_date_local=date(2020, 1, 1),
            end_date_local=date(2024, 12, 31),
        )

        assert config.station_ids == ["KLGA"]
        assert config.split.train_frac == 0.70

    def test_invalid_date_range(self) -> None:
        """Start after end should raise."""
        from tempdata.eval.config import EvalConfig

        with pytest.raises(ValueError, match="start_date_local.*must be before"):
            EvalConfig(
                run_name="test",
                station_ids=["KLGA"],
                start_date_local=date(2024, 1, 1),
                end_date_local=date(2020, 1, 1),
            )

    def test_json_serialization(self) -> None:
        """Config should roundtrip through JSON."""
        from tempdata.eval.config import EvalConfig

        config = EvalConfig(
            run_name="test_round_trip",
            station_ids=["KLGA", "KJFK"],
            start_date_local=date(2020, 1, 1),
            end_date_local=date(2024, 12, 31),
        )

        # To JSON and back
        json_str = config.to_json()
        restored = EvalConfig.from_json(json_str)

        assert restored.station_ids == config.station_ids
        assert restored.start_date_local == config.start_date_local
        assert restored.run_name == config.run_name


class TestSplits:
    """Tests for time-based splitting."""

    def test_static_split_fractions(self) -> None:
        """Static split should respect fractions."""
        from tempdata.eval.splits import StaticSplit

        # Create sample data
        df = pd.DataFrame({
            "value": range(100),
            "date": pd.date_range("2020-01-01", periods=100),
        })

        split = StaticSplit(train_frac=0.70, val_frac=0.15, test_frac=0.15)
        train, val, test = split.split(df)

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_static_split_no_overlap(self) -> None:
        """Static split should have no overlap between sets."""
        from tempdata.eval.splits import StaticSplit

        df = pd.DataFrame({
            "idx": range(100),
            "date": pd.date_range("2020-01-01", periods=100),
        })

        split = StaticSplit(train_frac=0.70, val_frac=0.15, test_frac=0.15)
        train, val, test = split.split(df)

        train_idx = set(train["idx"])
        val_idx = set(val["idx"])
        test_idx = set(test["idx"])

        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

    def test_static_split_temporal_order(self) -> None:
        """Static split should maintain temporal order (no leakage)."""
        from tempdata.eval.splits import StaticSplit

        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100),
        })

        split = StaticSplit(train_frac=0.70, val_frac=0.15, test_frac=0.15)
        train, val, test = split.split(df)

        # All train dates should be before val dates
        assert train["date"].max() < val["date"].min()
        # All val dates should be before test dates
        assert val["date"].max() < test["date"].min()


class TestModels:
    """Tests for forecast models."""

    @pytest.fixture
    def sample_train_data(self) -> pd.DataFrame:
        """Create sample training data."""
        n = 50
        rng = np.random.default_rng(42)

        return pd.DataFrame({
            "tmax_pred_f": 70 + rng.normal(0, 5, n),
            "sin_doy": rng.uniform(-1, 1, n),
            "cos_doy": rng.uniform(-1, 1, n),
            "bias_7d": rng.normal(0, 2, n),
            "bias_14d": rng.normal(0, 2, n),
            "tmax_actual_f": 70 + rng.normal(0, 5, n),
        })

    def test_passthrough_forecaster(self, sample_train_data: pd.DataFrame) -> None:
        """PassthroughForecaster should return tmax_pred_f."""
        from tempdata.eval.models import PassthroughForecaster

        model = PassthroughForecaster()
        model.fit(sample_train_data)

        predictions = model.predict_mu(sample_train_data)

        np.testing.assert_array_equal(
            predictions, sample_train_data["tmax_pred_f"].values
        )

    def test_ridge_forecaster(self, sample_train_data: pd.DataFrame) -> None:
        """RidgeForecaster should fit and predict."""
        pytest.importorskip("sklearn")
        from tempdata.eval.models import RidgeForecaster

        model = RidgeForecaster(alpha=1.0)
        model.fit(sample_train_data)

        predictions = model.predict_mu(sample_train_data)

        assert len(predictions) == len(sample_train_data)
        assert model.coef_ is not None
        assert len(model.coef_) == 5  # 5 features


class TestUncertainty:
    """Tests for uncertainty calibration."""

    @pytest.fixture
    def sample_data_with_residuals(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Create sample data with residuals."""
        n = 100
        rng = np.random.default_rng(42)

        df = pd.DataFrame({
            "lead_hours": rng.choice([24, 48, 72], n),
        })
        residuals = rng.normal(0, 3, n)

        return df, residuals

    def test_global_sigma(
        self, sample_data_with_residuals: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        """GlobalSigma should compute single sigma."""
        from tempdata.eval.uncertainty import GlobalSigma

        df, residuals = sample_data_with_residuals

        model = GlobalSigma(sigma_floor=1.0)
        model.fit(df, residuals)

        sigmas = model.predict_sigma(df)

        assert len(sigmas) == len(df)
        assert (sigmas == sigmas[0]).all()  # All same value
        assert sigmas[0] >= 1.0  # Floor applied

    def test_bucketed_sigma(
        self, sample_data_with_residuals: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        """BucketedSigma should compute different sigmas by bucket."""
        from tempdata.eval.uncertainty import BucketedSigma

        df, residuals = sample_data_with_residuals

        model = BucketedSigma(
            buckets=[(0, 36), (36, 60), (60, 100)],
            sigma_floor=1.0,
        )
        model.fit(df, residuals)

        sigmas = model.predict_sigma(df)

        assert len(sigmas) == len(df)


class TestMetrics:
    """Tests for metrics computation."""

    def test_forecast_metrics(self) -> None:
        """Forecast metrics should compute correctly."""
        from tempdata.eval.metrics import compute_forecast_metrics

        predictions_df = pd.DataFrame({
            "y_pred_f": [70.0, 75.0, 80.0],
            "y_true_f": [72.0, 73.0, 78.0],
        })

        metrics = compute_forecast_metrics(predictions_df)

        assert metrics.n_samples == 3
        assert metrics.mae == pytest.approx(2.0)
        assert metrics.bias == pytest.approx(2 / 3)  # Mean of -2, 2, 2

    def test_calibration_metrics(self) -> None:
        """Calibration metrics should compute coverage correctly."""
        pytest.importorskip("scipy")
        from tempdata.eval.metrics import compute_calibration_metrics

        # Create data where actual is always at the predicted mean
        predictions_df = pd.DataFrame({
            "y_pred_f": [70.0, 75.0, 80.0],
            "y_true_f": [70.0, 75.0, 80.0],  # Perfect predictions
            "y_pred_sigma_f": [3.0, 3.0, 3.0],
        })

        metrics = compute_calibration_metrics(predictions_df)

        # With perfect predictions, all should be covered
        assert metrics.coverage_50 == 1.0
        assert metrics.coverage_80 == 1.0
        assert metrics.coverage_90 == 1.0


class TestSlicing:
    """Tests for metrics slicing."""

    def test_slice_by_month(self) -> None:
        """Slicing by month should work."""
        from tempdata.eval.slicing import compute_metrics_by_slice

        # Create data spanning multiple months
        df = pd.DataFrame({
            "y_pred_f": [70.0] * 60,
            "y_true_f": [72.0] * 30 + [68.0] * 30,  # Different errors by month
            "month": [1] * 30 + [7] * 30,
        })

        slices = compute_metrics_by_slice(df)

        assert "by_month" in slices
        assert "1" in slices["by_month"]
        assert "7" in slices["by_month"]

    def test_slice_by_season(self) -> None:
        """Slicing by season should categorize correctly."""
        from tempdata.eval.slicing import compute_metrics_by_slice

        df = pd.DataFrame({
            "y_pred_f": [70.0] * 40,
            "y_true_f": [72.0] * 40,
            "month": [1] * 10 + [4] * 10 + [7] * 10 + [10] * 10,
        })

        slices = compute_metrics_by_slice(df)

        assert "by_season" in slices
        assert "DJF" in slices["by_season"]
        assert "JJA" in slices["by_season"]
