import pandas as pd
import pytest
from src.backtest.executor import Executor


class TestExecutor:
    def test_no_trade_no_cost(self):
        ex = Executor(commission_rate=0.001)
        target = pd.Series({"A": 0.5, "B": 0.5})
        current = pd.Series({"A": 0.5, "B": 0.5})
        prices = pd.Series({"A": 100, "B": 200})
        result = ex.execute(target, current, prices, capital=1_000_000)
        assert result["total_cost"] == pytest.approx(0.0, abs=0.01)

    def test_trade_generates_commission(self):
        ex = Executor(commission_rate=0.001, min_commission=0)
        target = pd.Series({"A": 1.0, "B": 0.0})
        current = pd.Series({"A": 0.0, "B": 1.0})
        prices = pd.Series({"A": 100, "B": 200})
        result = ex.execute(target, current, prices, capital=1_000_000)
        expected_commission = 1_000_000 * 0.001 * 2  # both A and B trade 100%
        assert result["commission"] == pytest.approx(expected_commission, rel=1e-3)
