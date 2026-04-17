"""Tests for new features: accuracy bands, report module, walk-forward, xgboost CLI."""
from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ─── Accuracy Bands ───────────────────────────────────────────────


class TestAccuracyBandMetrics:
    def test_within_bands(self):
        from tempdata.eval.metrics import compute_accuracy_bands, AccuracyBandMetrics

        df = pd.DataFrame({
            "y_true_f": [70.0, 71.0, 72.0, 73.0, 74.0],
            "y_pred_f": [70.5, 71.2, 73.5, 73.1, 75.0],  # errors: 0.5, 0.2, 1.5, 0.1, 1.0
        })

        result = compute_accuracy_bands(df)
        assert isinstance(result, AccuracyBandMetrics)
        assert result.n_samples == 5

        # Within 1F: 70.5(0.5), 71.2(0.2), 73.1(0.1), 75.0(1.0) = 4/5 = 0.8
        assert result.within_1f == pytest.approx(0.8)

        # Within 2F: all 5 = 1.0
        assert result.within_2f == pytest.approx(1.0)

    def test_perfect_predictions(self):
        from tempdata.eval.metrics import compute_accuracy_bands

        df = pd.DataFrame({
            "y_true_f": [70.0, 71.0, 72.0],
            "y_pred_f": [70.0, 71.0, 72.0],
        })
        result = compute_accuracy_bands(df)
        assert result.within_1f == 1.0
        assert result.within_2f == 1.0
        assert result.within_3f == 1.0
        assert result.within_5f == 1.0

    def test_to_dict(self):
        from tempdata.eval.metrics import compute_accuracy_bands

        df = pd.DataFrame({
            "y_true_f": [70.0],
            "y_pred_f": [70.5],
        })
        result = compute_accuracy_bands(df)
        d = result.to_dict()
        assert "within_1f" in d
        assert "n_samples" in d
        assert d["n_samples"] == 1

    def test_in_eval_metrics(self):
        from tempdata.eval.metrics import (
            EvalMetrics,
            compute_accuracy_bands,
            compute_forecast_metrics,
        )

        df = pd.DataFrame({
            "y_true_f": [70.0, 71.0, 72.0],
            "y_pred_f": [70.5, 71.2, 73.5],
        })
        fm = compute_forecast_metrics(df)
        ab = compute_accuracy_bands(df)

        metrics = EvalMetrics(forecast=fm, accuracy_bands=ab)
        d = metrics.to_dict()
        assert "accuracy_bands" in d
        assert d["accuracy_bands"]["n_samples"] == 3


# ─── Report Module ────────────────────────────────────────────────


class TestReportModule:
    def test_load_single_model_run(self, tmp_path):
        from tempdata.report.report_daily_tmax import load_run_summary

        # Create a fake single-model run
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        (run_dir / "meta.json").write_text(json.dumps({
            "run_id": "test_001",
            "run_name": "Test Run",
            "timestamp_utc": "2026-01-01T00:00:00",
        }))
        (run_dir / "metrics.json").write_text(json.dumps({
            "forecast": {"mae": 1.5, "rmse": 2.0, "bias": 0.1, "r2": 0.95, "n_samples": 100},
            "accuracy_bands": {"within_1f": 0.6, "within_2f": 0.85, "n_samples": 100},
        }))
        (run_dir / "config.json").write_text(json.dumps({
            "model": {
                "type": "stacked",
                "features": ["tmax_pred_f", "lead_hours", "bias_7d"],
                "hyperparams": {"meta_alpha": 0.01, "stacking_splits": 5},
            },
            "station_ids": ["KLGA"],
        }))

        summary = load_run_summary(run_dir)
        assert summary["run_id"] == "test_001"
        assert summary["type"] == "single_model"
        assert summary["model_type"] == "stacked"
        assert summary["features"] == ["tmax_pred_f", "lead_hours", "bias_7d"]
        assert summary["model_hyperparams"]["meta_alpha"] == 0.01
        assert summary["metrics"]["forecast"]["mae"] == 1.5

    def test_load_multi_model_run(self, tmp_path):
        from tempdata.report.report_daily_tmax import load_run_summary

        run_dir = tmp_path / "multi_run"
        run_dir.mkdir()
        (run_dir / "meta.json").write_text(json.dumps({
            "run_id": "multi_001",
            "is_multi_model": True,
            "model_names": ["ridge", "persistence"],
        }))

        models_dir = run_dir / "models"
        for model in ["ridge", "persistence"]:
            md = models_dir / model
            md.mkdir(parents=True)
            (md / "metrics.json").write_text(json.dumps({
                "forecast": {"mae": 1.0 if model == "ridge" else 2.5, "rmse": 1.5, "bias": 0.0, "r2": 0.99, "n_samples": 50},
            }))

        summary = load_run_summary(run_dir)
        assert summary["type"] == "multi_model"
        assert "ridge" in summary["models"]
        assert "persistence" in summary["models"]

    def test_compare_runs(self, tmp_path):
        from tempdata.report.report_daily_tmax import compare_runs

        for i, mae in enumerate([1.5, 2.0]):
            rd = tmp_path / f"run_{i}"
            rd.mkdir()
            (rd / "meta.json").write_text(json.dumps({"run_id": f"run_{i}"}))
            (rd / "metrics.json").write_text(json.dumps({
                "forecast": {"mae": mae, "rmse": mae + 0.5, "bias": 0.0, "r2": 0.9, "n_samples": 100},
                "accuracy_bands": {"within_1f": 0.6, "within_2f": 0.8},
            }))
            (rd / "config.json").write_text(json.dumps({
                "model": {"type": "ridge"},
                "station_ids": ["KLGA"],
            }))

        df = compare_runs([tmp_path / "run_0", tmp_path / "run_1"])
        assert len(df) == 2
        assert df.iloc[0]["mae"] == 1.5  # sorted by MAE
        assert "within_1f_pct" in df.columns
        assert "feature_count" in df.columns


# ─── Walk-Forward Evaluation ──────────────────────────────────────


class TestWalkForward:
    def test_generate_folds(self):
        from tempdata.eval.splits import WalkForwardSplit

        df = pd.DataFrame({"x": range(100)})
        splitter = WalkForwardSplit(window_size=50, step_size=10)
        folds = splitter.generate_folds(df)

        # First fold: train[0:50], test[50:60]
        assert len(folds) == 5  # (100-50)/10 = 5
        assert len(folds[0][0]) == 50  # train
        assert len(folds[0][1]) == 10  # test

        # Last fold: train[40:90], test[90:100]
        assert len(folds[-1][0]) == 50
        assert len(folds[-1][1]) == 10

    def test_expanding_folds(self):
        from tempdata.eval.splits import WalkForwardSplit

        df = pd.DataFrame({"x": range(100)})
        splitter = WalkForwardSplit(window_size=50, step_size=10, expanding=True)
        folds = splitter.generate_folds(df)

        # First fold: train[0:50], test[50:60]
        assert len(folds[0][0]) == 50
        # Last fold: train[0:90], test[90:100] (expanding)
        assert len(folds[-1][0]) == 90


# ─── XGBoost Model ────────────────────────────────────────────────


class TestXGBoostModel:
    def test_create_xgboost_forecaster(self):
        from tempdata.eval.models import create_forecaster, XGBoostForecaster

        model = create_forecaster("xgboost")
        assert isinstance(model, XGBoostForecaster)

    def test_xgboost_fit_predict(self):
        from tempdata.eval.models import XGBoostForecaster

        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "tmax_pred_f": np.random.uniform(60, 90, n),
            "sin_doy": np.sin(np.linspace(0, 2 * np.pi, n)),
            "cos_doy": np.cos(np.linspace(0, 2 * np.pi, n)),
            "bias_7d": np.random.uniform(-2, 2, n),
            "bias_14d": np.random.uniform(-2, 2, n),
            "lead_hours": np.random.choice([24, 48, 72], n),
            "tmax_actual_f": np.random.uniform(60, 90, n),
        })

        model = XGBoostForecaster()
        model.fit(df)
        preds = model.predict_mu(df)
        assert len(preds) == n
        assert all(np.isfinite(preds))



# ─── LightGBM / CatBoost / Stacking ───────────────────────────────


class _FakeBooster:
    def __init__(self, **params):
        self.params = params
        self.mean_ = None

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


class TestTabularChallengers:
    def test_create_lightgbm_forecaster(self):
        from tempdata.eval.models import create_forecaster, LightGBMForecaster

        model = create_forecaster("lightgbm")
        assert isinstance(model, LightGBMForecaster)

    def test_create_catboost_forecaster(self):
        from tempdata.eval.models import create_forecaster, CatBoostForecaster

        model = create_forecaster("catboost")
        assert isinstance(model, CatBoostForecaster)

    def test_lightgbm_fit_predict_with_fake_module(self, monkeypatch):
        import sys
        import types

        from tempdata.eval.models import LightGBMForecaster

        monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace(LGBMRegressor=_FakeBooster))

        df = pd.DataFrame({
            "tmax_pred_f": [70.0, 71.0, 72.0, 73.0],
            "sin_doy": [0.1, 0.2, 0.3, 0.4],
            "cos_doy": [0.9, 0.8, 0.7, 0.6],
            "bias_7d": [0.0, 0.1, 0.0, -0.1],
            "bias_14d": [0.0, 0.0, 0.1, 0.0],
            "lead_hours": [24, 24, 24, 24],
            "tmax_actual_f": [71.0, 72.0, 73.0, 74.0],
        })

        model = LightGBMForecaster()
        model.fit(df)
        preds = model.predict_mu(df)
        assert len(preds) == len(df)
        assert np.all(np.isfinite(preds))

    def test_catboost_fit_predict_with_fake_module(self, monkeypatch):
        import sys
        import types

        from tempdata.eval.models import CatBoostForecaster

        monkeypatch.setitem(sys.modules, "catboost", types.SimpleNamespace(CatBoostRegressor=_FakeBooster))

        df = pd.DataFrame({
            "tmax_pred_f": [70.0, 71.0, 72.0, 73.0],
            "sin_doy": [0.1, 0.2, 0.3, 0.4],
            "cos_doy": [0.9, 0.8, 0.7, 0.6],
            "bias_7d": [0.0, 0.1, 0.0, -0.1],
            "bias_14d": [0.0, 0.0, 0.1, 0.0],
            "lead_hours": [24, 24, 24, 24],
            "tmax_actual_f": [71.0, 72.0, 73.0, 74.0],
        })

        model = CatBoostForecaster()
        model.fit(df)
        preds = model.predict_mu(df)
        assert len(preds) == len(df)
        assert np.all(np.isfinite(preds))

    def test_stacked_ensemble_fits_and_predicts(self):
        from tempdata.eval.models import StackedEnsembleForecaster

        class _ScaledForecaster:
            def __init__(self, scale):
                self.scale = scale

            def fit(self, df_train):
                return None

            def predict_mu(self, df):
                return df["x"].to_numpy(dtype=float) * self.scale

        df = pd.DataFrame({
            "x": np.arange(1, 31, dtype=float),
            "tmax_actual_f": np.arange(1, 31, dtype=float) * 3.0,
        })

        model = StackedEnsembleForecaster(
            [
                ("m1", lambda: _ScaledForecaster(1.0)),
                ("m2", lambda: _ScaledForecaster(2.0)),
            ],
            meta_alpha=0.0,
        )
        model.fit(df)
        preds = model.predict_mu(df)

        assert len(preds) == len(df)
        assert np.all(np.isfinite(preds))
        assert np.mean(np.abs(preds - df["tmax_actual_f"].to_numpy())) < 1e-2


# ─── Data Path Unification ────────────────────────────────────────


class TestDataPaths:
    def test_feature_path_cli_canonical(self, tmp_path):
        """Verify _load_feature_input finds CLI's canonical path."""
        from tempdata.eval.data import _load_feature_input

        # Create the CLI canonical path
        cli_path = tmp_path / "features" / "train_daily_tmax"
        cli_path.mkdir(parents=True)
        df = pd.DataFrame({"a": [1, 2, 3]})
        (cli_path / "KLGA.parquet").write_bytes(df.to_parquet(index=False))

        result = _load_feature_input(tmp_path, "KLGA", None)
        assert result is not None
        assert len(result) == 3

    def test_forecast_path_cli_canonical(self, tmp_path):
        """Verify _load_forecast_input finds CLI's canonical path."""
        from tempdata.eval.data import _load_forecast_input

        cli_path = tmp_path / "raw" / "daily_tmax_forecast" / "KLGA"
        cli_path.mkdir(parents=True)
        df = pd.DataFrame({"tmax_pred_f": [70.0, 71.0]})
        (cli_path / "historical.parquet").write_bytes(df.to_parquet(index=False))

        result = _load_forecast_input(tmp_path, "KLGA", None)
        assert len(result) == 2

    def test_truth_path_cli_canonical(self, tmp_path):
        """Verify _load_truth_input finds CLI's canonical path."""
        from tempdata.eval.data import _load_truth_input

        cli_path = tmp_path / "clean" / "daily_tmax"
        cli_path.mkdir(parents=True)
        df = pd.DataFrame({"tmax_f": [70.0, 71.0]})
        (cli_path / "KLGA.parquet").write_bytes(df.to_parquet(index=False))

        result = _load_truth_input(tmp_path, "KLGA", None)
        assert len(result) == 2
