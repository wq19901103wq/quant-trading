import numpy as np
import pandas as pd
import pytest
from quant_trading.models.validation import evaluate_regression, ic_score, rank_ic_score


class TestValidation:
    def test_evaluate_regression(self):
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        result = evaluate_regression(y_true, y_pred)
        assert "mse" in result
        assert "rmse" in result
        assert "mae" in result
        assert "r2" in result
        assert result["mse"] >= 0

    def test_ic_score(self):
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([2, 4, 6, 8, 10])
        ic = ic_score(y_true, y_pred)
        assert ic == pytest.approx(1.0, abs=1e-6)

    def test_rank_ic_score(self):
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([2, 4, 6, 8, 10])
        ric = rank_ic_score(y_true, y_pred)
        assert ric == pytest.approx(1.0, abs=1e-6)
