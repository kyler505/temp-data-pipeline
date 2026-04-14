from __future__ import annotations

from datetime import date

import pandas as pd


def _sample_forecast_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["KLGA"],
            "target_date_local": [pd.Timestamp("2024-01-02")],
            "tmax_pred_f": [50.0],
            "lead_hours": [24],
        }
    )


def _sample_truth_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["KLGA"],
            "date_local": [pd.Timestamp("2024-01-02")],
            "tmax_f": [51.0],
            "coverage_hours": [24],
        }
    )


def _sample_feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["KLGA"],
            "issue_time_utc": [pd.Timestamp("2024-01-01T00:00:00Z")],
            "target_date_local": [pd.Timestamp("2024-01-02")],
            "tmax_pred_f": [50.0],
            "lead_hours": [24],
            "forecast_source": ["openmeteo"],
            "sin_doy": [0.1],
            "cos_doy": [0.9],
            "month": [1],
            "bias_7d": [0.0],
            "bias_14d": [0.0],
            "bias_30d": [0.0],
            "rmse_14d": [1.0],
            "rmse_30d": [1.0],
            "sigma_lead": [1.0],
            "tmax_actual_f": [51.0],
        }
    )


def test_load_eval_inputs_uses_canonical_paths(tmp_path):
    from tempdata.eval.config import EvalConfig
    from tempdata.eval.data import load_eval_inputs

    station = "KLGA"
    forecast_dir = tmp_path / "clean" / "forecasts" / "openmeteo" / station
    truth_dir = tmp_path / "clean" / "daily_tmax" / station
    feature_dir = tmp_path / "train" / "daily_tmax" / station
    forecast_dir.mkdir(parents=True)
    truth_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)

    _sample_forecast_df().to_parquet(forecast_dir / "historical.parquet", index=False)
    _sample_truth_df().to_parquet(truth_dir / "2024.parquet", index=False)
    _sample_feature_df().to_parquet(feature_dir / "train_daily_tmax.parquet", index=False)

    config = EvalConfig(
        run_name="test",
        station_ids=[station],
        start_date_local=date(2024, 1, 1),
        end_date_local=date(2024, 1, 31),
    )

    forecast_df, truth_df, feature_df = load_eval_inputs(config, data_dir=tmp_path)

    assert len(forecast_df) == 1
    assert len(truth_df) == 1
    assert feature_df is not None
    assert len(feature_df) == 1


def test_cli_eval_builds_valid_config_and_loads_inputs(monkeypatch):
    from tempdata import cli

    captured: dict[str, object] = {}

    def fake_load_eval_inputs(config, **kwargs):
        captured["config"] = config
        captured["load_kwargs"] = kwargs
        return _sample_forecast_df(), _sample_truth_df(), _sample_feature_df()

    def fake_run_evaluation(config, forecast_df, truth_df, feature_df, run_id):
        captured["run_config"] = config
        captured["forecast_df"] = forecast_df
        captured["truth_df"] = truth_df
        captured["feature_df"] = feature_df
        captured["run_id"] = run_id

    monkeypatch.setattr(cli, "load_eval_inputs", fake_load_eval_inputs)
    monkeypatch.setattr(cli, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(cli, "generate_run_id", lambda: "run-123")

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "eval",
            "--station",
            "KLGA",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--data-dir",
            "data",
        ]
    )

    args.func(args)

    config = captured["config"]
    assert config.station_ids == ["KLGA"]
    assert config.start_date_local == date(2024, 1, 1)
    assert config.end_date_local == date(2024, 1, 31)
    assert config.split.type == "static"
    assert captured["run_id"] == "run-123"
    assert captured["load_kwargs"] == {
        "data_dir": "data",
        "forecast_file": None,
        "truth_file": None,
        "feature_file": None,
    }


def test_cli_eval_loads_config_file(monkeypatch, tmp_path):
    from tempdata import cli
    from tempdata.eval.config import EvalConfig

    config = EvalConfig(
        run_name="cfg",
        station_ids=["KLGA"],
        start_date_local=date(2024, 1, 1),
        end_date_local=date(2024, 1, 31),
    )
    config_path = tmp_path / "eval.json"
    config.save(config_path)

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "load_eval_inputs",
        lambda loaded_config, **kwargs: (
            captured.setdefault("config", loaded_config) and _sample_forecast_df(),
            _sample_truth_df(),
            _sample_feature_df(),
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda loaded_config, forecast_df, truth_df, feature_df, run_id: captured.update(
            {"run_config": loaded_config, "run_id": run_id}
        ),
    )
    monkeypatch.setattr(cli, "generate_run_id", lambda: "cfg-run")

    parser = cli.build_parser()
    args = parser.parse_args(["eval", "--config", str(config_path)])
    args.func(args)

    loaded_config = captured["config"]
    assert loaded_config.run_name == "cfg"
    assert loaded_config.station_ids == ["KLGA"]
    assert captured["run_id"] == "cfg-run"


def test_load_eval_data_normalizes_datetime_target_dates():
    from tempdata.eval.config import EvalConfig
    from tempdata.eval.data import load_eval_data

    feature_df = _sample_feature_df()
    feature_df["target_date_local"] = pd.to_datetime(feature_df["target_date_local"])

    config = EvalConfig(
        run_name="normalize",
        station_ids=["KLGA"],
        start_date_local=date(2024, 1, 1),
        end_date_local=date(2024, 1, 31),
    )

    dataset = load_eval_data(
        config=config,
        forecast_df=_sample_forecast_df(),
        truth_df=_sample_truth_df(),
        feature_df=feature_df,
    )

    assert len(dataset.full) == 1
    assert dataset.full.iloc[0]["target_date_local"] == date(2024, 1, 2)
