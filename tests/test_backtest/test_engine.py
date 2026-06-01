import pandas as pd
import numpy as np
from src.backtest.engine import BacktestEngine
from src.portfolio.strategy import TopKStrategy
from src.backtest.executor import Executor


class TestBacktestEngine:
    def test_basic_run(self):
        dates = pd.date_range("2020-01-01", periods=5)
        predictions = pd.DataFrame({
            "A": [0.1, 0.2, 0.3, 0.2, 0.1],
            "B": [0.2, 0.1, 0.1, 0.3, 0.2],
        }, index=dates)
        prices = pd.DataFrame({
            "A": [100, 101, 102, 103, 104],
            "B": [200, 198, 202, 201, 205],
        }, index=dates)
        engine = BacktestEngine(
            strategy=TopKStrategy(k=1),
            executor=Executor(commission_rate=0.0),
            initial_capital=1_000_000,
        )
        result = engine.run(predictions, prices)
        assert "metrics" in result
        assert "returns" in result
        assert "portfolio_values" in result
        assert result["portfolio_values"].iloc[0] == 1_000_000

    def test_returns_series_length(self):
        dates = pd.date_range("2020-01-01", periods=5)
        predictions = pd.DataFrame({"A": [0.1] * 5, "B": [0.2] * 5}, index=dates)
        prices = pd.DataFrame({"A": [100] * 5, "B": [200] * 5}, index=dates)
        engine = BacktestEngine(
            strategy=TopKStrategy(k=1),
            executor=Executor(commission_rate=0.0),
            initial_capital=1_000_000,
        )
        result = engine.run(predictions, prices)
        assert len(result["returns"]) == 5
