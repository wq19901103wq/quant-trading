import pandas as pd
import numpy as np
import pytest
from src.metrics.core import calculate_metrics


class TestCalculateMetrics:
    def test_zero_returns(self):
        returns = pd.Series([0.0] * 100)
        metrics = calculate_metrics(returns)
        assert metrics["total_return"] == pytest.approx(0.0)
        assert metrics["sharpe_ratio"] == pytest.approx(0.0)

    def test_positive_returns(self):
        returns = pd.Series([0.001] * 252)
        metrics = calculate_metrics(returns)
        assert metrics["total_return"] > 0
        assert metrics["annualized_return"] > 0
        assert metrics["sharpe_ratio"] > 0

    def test_drawdown(self):
        returns = pd.Series([0.01, 0.01, -0.05, 0.01, 0.01])
        metrics = calculate_metrics(returns)
        assert metrics["max_drawdown"] < 0
