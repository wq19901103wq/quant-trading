import pandas as pd
import numpy as np
import pytest
from quant_trading.factors.evaluator import FactorEvaluator


class TestFactorEvaluator:
    def test_ic_perfect_correlation(self):
        ev = FactorEvaluator()
        factor = pd.Series([1, 2, 3, 4, 5])
        forward = pd.Series([2, 4, 6, 8, 10])
        result = ev.evaluate(factor, forward)
        assert result["ic"] == pytest.approx(1.0, abs=1e-6)

    def test_ic_zero_correlation(self):
        ev = FactorEvaluator()
        factor = pd.Series([1, 2, 3, 4, 5])
        forward = pd.Series([5, 4, 3, 2, 1])
        result = ev.evaluate(factor, forward)
        assert result["ic"] == pytest.approx(-1.0, abs=1e-6)
