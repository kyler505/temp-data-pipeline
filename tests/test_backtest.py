"""Tests for backtest module.

Key tests:
- Config validation
- Data loading and joining
- Model fitting and prediction
- Bin probability calculation
- Strategy order generation
- Simulator execution and settlement
- Metrics computation
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""
    
    def test_valid_config(self) -> None:
        """Valid config should pass validation."""
        from tempdata.backtest.config import BacktestConfig
        
        config = BacktestConfig(
            station_ids=["KLGA"],
            start_date_local=date(2020, 1, 1),
            end_date_local=date(2024, 12, 31),
        )
        
        assert config.station_ids == ["KLGA"]
        assert config.train_frac == 0.70
        assert config.test_frac == pytest.approx(0.15)
    
    def test_invalid_date_range(self) -> None:
        """Start after end should raise."""
        from tempdata.backtest.config import BacktestConfig
        
        with pytest.raises(ValueError, match="start_date_local.*must be before"):
            BacktestConfig(
                station_ids=["KLGA"],
                start_date_local=date(2024, 1, 1),
                end_date_local=date(2020, 1, 1),
            )
    
    def test_invalid_split_fractions(self) -> None:
        """Split fractions >= 1 should raise."""
        from tempdata.backtest.config import BacktestConfig
        
        with pytest.raises(ValueError, match="train_frac.*val_frac"):
            BacktestConfig(
                station_ids=["KLGA"],
                start_date_local=date(2020, 1, 1),
                end_date_local=date(2024, 12, 31),
                train_frac=0.8,
                val_frac=0.3,  # Sum > 1
            )
    
    def test_json_serialization(self) -> None:
        """Config should roundtrip through JSON."""
        from tempdata.backtest.config import BacktestConfig
        
        config = BacktestConfig(
            station_ids=["KLGA", "KJFK"],
            start_date_local=date(2020, 1, 1),
            end_date_local=date(2024, 12, 31),
            edge_min=0.05,
        )
        
        # To JSON and back
        json_str = config.to_json()
        restored = BacktestConfig.from_json(json_str)
        
        assert restored.station_ids == config.station_ids
        assert restored.start_date_local == config.start_date_local
        assert restored.edge_min == config.edge_min


class TestBins:
    """Tests for bin definitions and probability calculation."""
    
    def test_binset_creation(self) -> None:
        """BinSet should validate bins."""
        from tempdata.backtest.bins import BinSet
        
        bins = BinSet([
            (float("-inf"), 50.0),
            (50.0, 70.0),
            (70.0, float("inf")),
        ])
        
        assert len(bins) == 3
        assert bins[0].label() == "< 50.0°F"
        assert bins[1].label() == "50.0-70.0°F"
        assert bins[2].label() == ">= 70.0°F"
    
    def test_binset_gap_detection(self) -> None:
        """BinSet should detect gaps between bins."""
        from tempdata.backtest.bins import BinSet
        
        with pytest.raises(ValueError, match="Gap between bins"):
            BinSet([
                (float("-inf"), 50.0),
                (60.0, float("inf")),  # Gap at 50-60
            ])
    
    def test_find_bin(self) -> None:
        """find_bin should return correct bin."""
        from tempdata.backtest.bins import BinSet
        
        bins = BinSet([
            (float("-inf"), 50.0),
            (50.0, 70.0),
            (70.0, float("inf")),
        ])
        
        assert bins.find_bin(45.0).bin_id == 0
        assert bins.find_bin(55.0).bin_id == 1
        assert bins.find_bin(80.0).bin_id == 2
        assert bins.find_bin(50.0).bin_id == 1  # Inclusive lower bound
    
    def test_bin_probabilities_sum_to_one(self) -> None:
        """Bin probabilities should sum to 1."""
        pytest.importorskip("scipy")
        from tempdata.backtest.bins import BinSet, compute_bin_probabilities
        
        bins = BinSet([
            (float("-inf"), 50.0),
            (50.0, 70.0),
            (70.0, float("inf")),
        ])
        
        mu = np.array([60.0, 55.0, 75.0])
        sigma = np.array([5.0, 10.0, 3.0])
        
        probs = compute_bin_probabilities(mu, sigma, bins)
        
        assert probs.shape == (3, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    
    def test_bin_probabilities_concentration(self) -> None:
        """With small sigma, probability should concentrate in correct bin."""
        pytest.importorskip("scipy")
        from tempdata.backtest.bins import BinSet, compute_bin_probabilities
        
        bins = BinSet([
            (float("-inf"), 50.0),
            (50.0, 70.0),
            (70.0, float("inf")),
        ])
        
        # Very tight distribution centered at 60
        mu = np.array([60.0])
        sigma = np.array([0.1])
        
        probs = compute_bin_probabilities(mu, sigma, bins)
        
        # Should be almost all in bin 1 (50-70)
        assert probs[0, 1] > 0.99


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
        from tempdata.backtest.models import PassthroughForecaster
        
        model = PassthroughForecaster()
        model.fit(sample_train_data)
        
        predictions = model.predict_mu(sample_train_data)
        
        np.testing.assert_array_equal(
            predictions, sample_train_data["tmax_pred_f"].values
        )
    
    def test_ridge_forecaster(self, sample_train_data: pd.DataFrame) -> None:
        """RidgeForecaster should fit and predict."""
        pytest.importorskip("sklearn")
        from tempdata.backtest.models import RidgeForecaster
        
        model = RidgeForecaster(alpha=1.0)
        model.fit(sample_train_data)
        
        predictions = model.predict_mu(sample_train_data)
        
        assert len(predictions) == len(sample_train_data)
        assert model.coef_ is not None
        assert len(model.coef_) == 5  # 5 features
    
    def test_ridge_handles_nan(self, sample_train_data: pd.DataFrame) -> None:
        """RidgeForecaster should handle NaN in features."""
        pytest.importorskip("sklearn")
        from tempdata.backtest.models import RidgeForecaster
        
        # Add some NaN
        sample_train_data.loc[0, "bias_7d"] = np.nan
        sample_train_data.loc[1, "bias_14d"] = np.nan
        
        model = RidgeForecaster(alpha=1.0, handle_missing="fill_zero")
        model.fit(sample_train_data)
        
        predictions = model.predict_mu(sample_train_data)
        assert not np.isnan(predictions).any()


class TestCalibration:
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
        from tempdata.backtest.calibration import GlobalSigma
        
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
        from tempdata.backtest.calibration import BucketedSigma
        
        df, residuals = sample_data_with_residuals
        
        model = BucketedSigma(
            buckets=[(0, 36), (36, 60), (60, 100)],
            sigma_floor=1.0,
        )
        model.fit(df, residuals)
        
        sigmas = model.predict_sigma(df)
        
        assert len(sigmas) == len(df)
        
        # Check bucket sigmas are different (or same if insufficient data)
        bucket_sigmas = model.get_bucket_sigmas()
        assert len(bucket_sigmas) == 3


class TestStrategy:
    """Tests for trading strategies."""
    
    def test_edge_strategy_no_trade_low_edge(self) -> None:
        """EdgeStrategy should not trade when edge is low."""
        from tempdata.backtest.strategy import EdgeStrategy
        
        strategy = EdgeStrategy(edge_min=0.05)
        
        row = pd.Series({"station_id": "KLGA"})
        prices = {0: 0.50, 1: 0.50}
        bin_probs = np.array([0.52, 0.48])  # Edge is only 0.02
        
        orders = strategy.generate_orders(
            row=row,
            prices=prices,
            bin_probs=bin_probs,
            bankroll=10000,
            exposure={},
        )
        
        assert len(orders) == 0
    
    def test_edge_strategy_trades_high_edge(self) -> None:
        """EdgeStrategy should trade when edge exceeds threshold."""
        from tempdata.backtest.strategy import EdgeStrategy, Side
        
        strategy = EdgeStrategy(edge_min=0.05, max_per_market_pct=0.02)
        
        row = pd.Series({"station_id": "KLGA"})
        prices = {0: 0.40, 1: 0.60}
        bin_probs = np.array([0.50, 0.50])  # Edge is 0.10 on bin 0
        
        orders = strategy.generate_orders(
            row=row,
            prices=prices,
            bin_probs=bin_probs,
            bankroll=10000,
            exposure={},
        )
        
        assert len(orders) == 1
        assert orders[0].bin_id == 0
        assert orders[0].side == Side.BUY
        assert orders[0].edge == pytest.approx(0.10)
    
    def test_edge_strategy_respects_exposure_limit(self) -> None:
        """EdgeStrategy should respect max total exposure."""
        from tempdata.backtest.strategy import EdgeStrategy
        
        strategy = EdgeStrategy(edge_min=0.01, max_total_pct=0.10)
        
        row = pd.Series({"station_id": "KLGA"})
        prices = {0: 0.30, 1: 0.70}  # Both have edge
        bin_probs = np.array([0.50, 0.50])
        
        orders = strategy.generate_orders(
            row=row,
            prices=prices,
            bin_probs=bin_probs,
            bankroll=10000,
            exposure={0: 500},  # Already have some exposure
        )
        
        # Total new exposure should be limited
        total_new = sum(o.size for o in orders)
        assert total_new <= 1000 - 500  # max_total - existing


class TestSimulator:
    """Tests for trade simulation."""
    
    @pytest.fixture
    def sample_backtest_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create sample train and test data."""
        rng = np.random.default_rng(42)
        
        def make_df(n: int, start_date: str) -> pd.DataFrame:
            dates = pd.date_range(start_date, periods=n, freq="D")
            return pd.DataFrame({
                "station_id": "KLGA",
                "issue_time_utc": pd.to_datetime(dates).tz_localize("UTC"),
                "target_date_local": dates,
                "tmax_pred_f": 70 + rng.normal(0, 5, n),
                "tmax_actual_f": 70 + rng.normal(0, 5, n),
                "lead_hours": 28,
                "sin_doy": np.sin(2 * np.pi * np.arange(n) / 365),
                "cos_doy": np.cos(2 * np.pi * np.arange(n) / 365),
                "bias_7d": rng.normal(0, 1, n),
                "bias_14d": rng.normal(0, 1, n),
            })
        
        train_df = make_df(100, "2020-01-01")
        test_df = make_df(50, "2020-04-11")
        
        return train_df, test_df
    
    def test_simulator_basic_run(
        self, sample_backtest_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Simulator should run without errors."""
        pytest.importorskip("sklearn")
        pytest.importorskip("scipy")
        
        from tempdata.backtest.bins import BinSet
        from tempdata.backtest.calibration import GlobalSigma
        from tempdata.backtest.models import PassthroughForecaster
        from tempdata.backtest.pricing import FixedSpreadPriceProvider
        from tempdata.backtest.simulator import run_backtest
        from tempdata.backtest.strategy import EdgeStrategy
        
        train_df, test_df = sample_backtest_data
        
        bins = BinSet([
            (float("-inf"), 60.0),
            (60.0, 80.0),
            (80.0, float("inf")),
        ])
        
        trades_df, daily_df, predictions_df = run_backtest(
            df_train=train_df,
            df_test=test_df,
            forecaster=PassthroughForecaster(),
            uncertainty_model=GlobalSigma(),
            strategy=EdgeStrategy(edge_min=0.01),
            price_provider=FixedSpreadPriceProvider(),
            bins=bins,
            slippage=0.01,
            initial_bankroll=10000.0,
            verbose=False,
        )
        
        # Should have predictions for all test rows
        assert len(predictions_df) == len(test_df)
        
        # Predictions should have mu and sigma
        assert "mu_f" in predictions_df.columns
        assert "sigma_f" in predictions_df.columns
        
        # Should have bin probability columns
        assert "p_bin_0" in predictions_df.columns


class TestMetrics:
    """Tests for metrics computation."""
    
    def test_forecast_metrics(self) -> None:
        """Forecast metrics should compute correctly."""
        from tempdata.backtest.metrics import compute_forecast_metrics
        
        predictions_df = pd.DataFrame({
            "mu_f": [70.0, 75.0, 80.0],
            "sigma_f": [3.0, 3.0, 3.0],
            "tmax_actual_f": [72.0, 73.0, 78.0],
        })
        
        metrics = compute_forecast_metrics(predictions_df)
        
        assert metrics.n_samples == 3
        assert metrics.mae == pytest.approx(2.0)
        assert metrics.bias == pytest.approx(2 / 3)  # Mean of -2, 2, 2
    
    def test_trading_metrics_empty(self) -> None:
        """Trading metrics should handle empty trades."""
        from tempdata.backtest.metrics import compute_trading_metrics
        
        trades_df = pd.DataFrame(columns=[
            "trade_id", "pnl", "is_settled", "size", "edge",
            "target_date_local",
        ])
        daily_df = pd.DataFrame(columns=["total_pnl", "target_date_local"])
        
        metrics = compute_trading_metrics(trades_df, daily_df, 10000.0)
        
        assert metrics.n_trades == 0
        assert metrics.total_pnl == 0.0
        assert metrics.win_rate == 0.0
    
    def test_trading_metrics_with_trades(self) -> None:
        """Trading metrics should compute correctly."""
        from tempdata.backtest.metrics import compute_trading_metrics
        
        trades_df = pd.DataFrame({
            "trade_id": [1, 2, 3],
            "pnl": [100.0, -50.0, 75.0],
            "is_settled": [True, True, True],
            "size": [200.0, 200.0, 200.0],
            "edge": [0.05, 0.03, 0.04],
            "target_date_local": pd.to_datetime([
                "2020-01-01", "2020-01-02", "2020-01-03"
            ]),
        })
        
        daily_df = pd.DataFrame({
            "target_date_local": pd.to_datetime([
                "2020-01-01", "2020-01-02", "2020-01-03"
            ]),
            "total_pnl": [100.0, -50.0, 75.0],
        })
        
        metrics = compute_trading_metrics(trades_df, daily_df, 10000.0)
        
        assert metrics.n_trades == 3
        assert metrics.total_pnl == pytest.approx(125.0)
        assert metrics.win_rate == pytest.approx(2 / 3)


class TestDataLoading:
    """Tests for data loading and joining."""
    
    @pytest.fixture
    def sample_forecast_truth(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create sample forecast and truth data."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        
        forecast_df = pd.DataFrame({
            "station_id": "KLGA",
            "lat": 40.78,
            "lon": -73.87,
            "issue_time_utc": pd.to_datetime(dates - pd.Timedelta(days=1)).tz_localize("UTC"),
            "target_date_local": dates,
            "tmax_pred_c": 20.0 + np.arange(10),
            "tmax_pred_f": 68.0 + np.arange(10) * 1.8,
            "lead_hours": 28,
            "model": "openmeteo",
            "source": "openmeteo",
            "ingested_at_utc": pd.Timestamp.now(tz="UTC"),
        })
        
        truth_df = pd.DataFrame({
            "date_local": dates,
            "station_id": "KLGA",
            "tmax_c": 19.0 + np.arange(10),
            "tmax_f": 66.2 + np.arange(10) * 1.8,
            "coverage_hours": 24,
            "source": "noaa",
            "qc_flags": 0,
            "updated_at_utc": pd.Timestamp.now(tz="UTC"),
        })
        
        return forecast_df, truth_df
    
    def test_data_overlap_check(
        self, sample_forecast_truth: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """check_data_overlap should report correct overlap."""
        from tempdata.backtest.data import check_data_overlap
        
        forecast_df, truth_df = sample_forecast_truth
        
        overlap = check_data_overlap(forecast_df, truth_df)
        
        assert overlap["overlap_days"] == 10
        assert "KLGA" in overlap["common_stations"]
