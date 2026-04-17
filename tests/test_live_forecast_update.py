"""Tests for live forecast script updates."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_live_forecast_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "live_forecast.py"
    spec = importlib.util.spec_from_file_location("live_forecast", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLiveForecastFeatureEngineering:
    def test_build_live_features_uses_updated_feature_set(self):
        live_forecast = _load_live_forecast_module()

        historical_df = pd.DataFrame(
            {
                "target_date_local": pd.to_datetime([
                    "2026-04-10",
                    "2026-04-11",
                    "2026-04-12",
                    "2026-04-13",
                ]),
                "tmax_pred_f": [70.0, 71.0, 72.0, 73.0],
                "tmax_actual_f": [71.0, 72.0, 73.0, 74.0],
                "bias_7d": [0.1, 0.2, 0.1, 0.0],
                "bias_14d": [0.1, 0.1, 0.0, -0.1],
                "bias_30d": [0.0, 0.0, 0.1, 0.1],
                "rmse_14d": [1.1, 1.0, 0.9, 0.8],
                "rmse_30d": [1.3, 1.2, 1.1, 1.0],
                "sigma_lead": [1.5, 1.4, 1.3, 1.2],
            }
        )
        forecast_row = pd.Series(
            {
                "station_id": "KLGA",
                "target_date_local": pd.Timestamp("2026-04-14"),
                "tmax_pred_f": 80.0,
                "lead_hours": 11,
            }
        )

        features = live_forecast.build_live_features(forecast_row, historical_df)

        assert "sin_doy" not in features.columns
        assert "cos_doy" not in features.columns
        assert features.loc[0, "rmse_30d"] == 1.0
        assert features.loc[0, "bias_30d"] == 1.0
        assert features.loc[0, "month"] == 4
        expected_cols = {
            "station_id",
            "target_date_local",
            "tmax_pred_f",
            "lead_hours",
            "month",
            "bias_7d",
            "bias_14d",
            "bias_30d",
            "rmse_14d",
            "rmse_30d",
            "sigma_lead",
        }
        assert expected_cols.issubset(set(features.columns))


class TestLiveForecastReporting:
    def test_accuracy_report_uses_model_label(self, tmp_path, capsys):
        live_forecast = _load_live_forecast_module()

        log_path = tmp_path / "predictions.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    '{"target_date": "2026-04-14", "actual_f": 84.6, "raw_forecast_f": 84.6, "ridge_prediction_f": 84.0, "raw_error_f": 0.0, "error_f": -0.6, "model_type": "stacked"}',
                    '{"target_date": "2026-04-15", "actual_f": 88.2, "raw_forecast_f": 88.9, "ridge_prediction_f": 88.3, "raw_error_f": 0.7, "error_f": 0.1, "model_type": "stacked"}',
                ]
            )
        )

        live_forecast.print_accuracy_report(log_path)
        out = capsys.readouterr().out

        assert "stacked" in out.lower()
        assert "MAE" in out
        assert "Predictions: 2 scored / 2 total" in out
