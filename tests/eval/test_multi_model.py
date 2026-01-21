
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import pytest
from datetime import date

from tempdata.eval.config import EvalConfig, ModelConfig, SplitConfig
from tempdata.eval.runner import run_multi_model_evaluation, MultiModelEvalResult
from tempdata.eval.report import load_multi_model_run, list_runs

@pytest.fixture
def temp_run_dir():
    """Create a temp directory for runs."""
    d = tempfile.mkdtemp()
    path = Path(d)
    yield path
    shutil.rmtree(d)

@pytest.fixture
def simple_data():
    """Generate dummy forecast and truth data."""
    dates = pd.date_range("2024-01-01", "2024-01-10")
    forecast_df = pd.DataFrame({
        "station_id": ["TEST"] * 10,
        "target_date_local": dates,
        "tmax_pred_f": [50.0 + i for i in range(10)],
        "lead_hours": [24] * 10
    })
    truth_df = pd.DataFrame({
        "station_id": ["TEST"] * 10,
        "date_local": dates,
        "tmax_f": [51.0 + i for i in range(10)]
    })
    return forecast_df, truth_df

def test_run_multi_model_evaluation(temp_run_dir, simple_data):
    forecast_df, truth_df = simple_data

    # Define two simple configs
    config1 = EvalConfig(
        run_name="model_a",
        station_ids=["TEST"],
        start_date_local=date(2024, 1, 1),
        end_date_local=date(2024, 1, 10),
        split=SplitConfig(type="static", train_frac=0.5, val_frac=0.25, test_frac=0.25),
        model=ModelConfig(type="persistence")
    )

    config2 = EvalConfig(
        run_name="model_b",
        station_ids=["TEST"],
        start_date_local=date(2024, 1, 1),
        end_date_local=date(2024, 1, 10),
        split=SplitConfig(type="static", train_frac=0.5, val_frac=0.25, test_frac=0.25),
        model=ModelConfig(type="ridge")
    )

    configs = {"Persistence": config1, "Ridge": config2}

    # Run evaluation
    result = run_multi_model_evaluation(
        configs=configs,
        forecast_df=forecast_df,
        truth_df=truth_df,
        output_dir=temp_run_dir,
        verbose=False
    )

    # Verify result structure
    assert isinstance(result, MultiModelEvalResult)
    assert result.run_id is not None
    assert "Persistence" in result.results
    assert "Ridge" in result.results
    assert "comparison" in result.__dict__ or hasattr(result, "comparison")

    # Verify file structure on disk
    run_path = result.run_path
    assert run_path.exists()
    assert (run_path / "meta.json").exists()
    assert (run_path / "comparison.json").exists()
    assert (run_path / "models" / "Persistence").is_dir()
    assert (run_path / "models" / "Ridge").is_dir()
    assert (run_path / "models" / "Persistence" / "metrics.json").exists()

    # Verify loading back
    loaded_data = load_multi_model_run(result.run_id, base_path=temp_run_dir)
    assert loaded_data["meta"]["run_id"] == result.run_id
    assert loaded_data["meta"]["is_multi_model"] is True
    assert "Persistence" in loaded_data["models"]
    assert "Ridge" in loaded_data["models"]

    # Verify list_runs detection
    runs = list_runs(base_path=temp_run_dir)
    assert len(runs) >= 1
    found = next((r for r in runs if r["run_id"] == result.run_id), None)
    assert found is not None
    assert found["is_multi_model"] is True
    assert "Persistence" in found["model_names"]
