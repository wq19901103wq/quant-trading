import pandas as pd
import numpy as np
import pytest
from quant_trading.models.cross_validation import TimeSeriesCV
from quant_trading.models.gbdt import LightGBMModel


class TestTimeSeriesCV:
    def test_split_structure(self):
        X = pd.DataFrame({"f1": range(100)})
        cv = TimeSeriesCV(n_splits=3)
        splits = cv.split(X)
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert train_idx.max() < test_idx.min()

    def test_train_size(self):
        X = pd.DataFrame({"f1": range(100)})
        cv = TimeSeriesCV(n_splits=2, train_size=20)
        splits = cv.split(X)
        for train_idx, test_idx in splits:
            assert len(train_idx) <= 20

    def test_cross_val_score(self):
        np.random.seed(42)
        X = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)})
        y = X["f1"] * 2 + np.random.randn(50) * 0.1
        cv = TimeSeriesCV(n_splits=2)
        model = LightGBMModel(params={"objective": "regression", "verbose": -1, "seed": 42})
        scores = cv.cross_val_score(model, X, y, metric_fn=lambda yt, yp: np.mean((yt - yp) ** 2))
        assert len(scores) == 2
        assert all(isinstance(s, (int, float)) for s in scores)
