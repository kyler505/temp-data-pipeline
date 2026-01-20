import numpy as np
import pandas as pd
import pytest
from tempdata.eval.models import KNNForecaster, PersistenceForecaster, create_forecaster


class TestPersistenceForecaster:
    def test_predict_standard(self):
        """Test standard persistence prediction using lag feature."""
        # Setup: lag1 is yesterday's tmax
        df = pd.DataFrame({
            "tmax_actual_f_lag1": [50.0, 55.0, 60.0],
            "tmax_pred_f": [100.0, 100.0, 100.0]  # Irrelevant for persistence
        })

        model = PersistenceForecaster()
        model.fit(df)  # Should do nothing
        preds = model.predict_mu(df)

        np.testing.assert_array_equal(preds, [50.0, 55.0, 60.0])

    def test_predict_fallback_logic(self):
        """Test fallback feature selection logic."""
        # Case 1: Multiple lag cols, picks smallest lag
        df = pd.DataFrame({
            "tmax_actual_f_lag1": [1.0],
            "tmax_actual_f_lag2": [2.0]
        })
        model = PersistenceForecaster()
        preds = model.predict_mu(df)
        assert preds[0] == 1.0

    def test_missing_lag_feature_raises(self):
        """Test error when no lag feature exists."""
        df = pd.DataFrame({"random_col": [1, 2, 3]})
        model = PersistenceForecaster()
        with pytest.raises(ValueError, match="Persistence requires a lag feature"):
            model.predict_mu(df)

    def test_factory_creation(self):
        model = create_forecaster("persistence")
        assert isinstance(model, PersistenceForecaster)


class TestKNNForecaster:
    def test_fit_predict(self):
        """Test basic fit and predict cycle."""
        # Train: 3 simple points
        # Day 1: Hot -> Hot output
        # Day 2: Cold -> Cold output
        # Day 3: Med -> Med output
        df_train = pd.DataFrame({
            "tmax_pred_f": [80.0, 30.0, 55.0],
            "lead_hours": [24, 24, 24],
            "sin_doy": [0, 0, 0],
            "cos_doy": [1, 1, 1],
            "tmax_actual_f": [82.0, 28.0, 56.0]
        })

        # Test: Query point close to Hot one
        df_test = pd.DataFrame({
            "tmax_pred_f": [81.0],
            "lead_hours": [24],
            "sin_doy": [0],
            "cos_doy": [1]
        })

        # 1 neighbor -> should pick closest (index 0)
        model = KNNForecaster(n_neighbors=1)
        model.fit(df_train)
        preds = model.predict_mu(df_test)

        assert preds[0] == 82.0

    def test_n_neighbors_averaging(self):
        """Test averaging of neighbors."""
        df_train = pd.DataFrame({
            "feature1": [1.0, 1.1, 100.0],
            "tmax_actual_f": [10.0, 20.0, 900.0]
        })

        df_test = pd.DataFrame({"feature1": [1.05]})

        model = KNNForecaster(n_neighbors=2, features=["feature1"])
        model.fit(df_train)
        preds = model.predict_mu(df_test)

        # Should average 10 and 20 -> 15
        assert preds[0] == 15.0

    def test_factory_creation(self):
        model = create_forecaster("knn")
        assert isinstance(model, KNNForecaster)
