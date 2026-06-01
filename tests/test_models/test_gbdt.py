import pandas as pd
import numpy as np
import pytest
from src.models.gbdt import LightGBMModel


class TestLightGBMModel:
    def test_fit_and_predict(self):
        np.random.seed(42)
        X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
        y = X["f1"] * 2 + X["f2"] * 0.5 + np.random.randn(100) * 0.1
        model = LightGBMModel(params={"objective": "regression", "verbose": -1, "seed": 42})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 100
        assert isinstance(preds, np.ndarray)

    def test_predict_before_fit_raises(self):
        model = LightGBMModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(pd.DataFrame({"f1": [1]}))

    def test_feature_importance(self):
        np.random.seed(42)
        X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
        y = X["f1"] * 2 + np.random.randn(100) * 0.1
        model = LightGBMModel(params={"objective": "regression", "verbose": -1, "seed": 42})
        model.fit(X, y)
        importance = model.feature_importance()
        assert len(importance) == 2
        assert "f1" in importance.index
