import pandas as pd
import pytest
from quant_trading.portfolio.strategy import TopKStrategy, RankWeightedStrategy, LongShortStrategy


class TestTopKStrategy:
    def test_top_k_weights_sum_to_one(self):
        preds = pd.Series({"A": 0.1, "B": 0.5, "C": 0.3, "D": 0.05})
        strategy = TopKStrategy(k=2)
        weights = strategy.generate_weights(preds)
        assert weights.sum() == pytest.approx(1.0)
        assert weights["B"] == pytest.approx(0.5)
        assert weights["C"] == pytest.approx(0.5)

    def test_zero_weights_for_non_top(self):
        preds = pd.Series({"A": 0.1, "B": 0.5, "C": 0.3})
        strategy = TopKStrategy(k=1)
        weights = strategy.generate_weights(preds)
        assert weights["A"] == 0.0
        assert weights["C"] == 0.0


class TestRankWeightedStrategy:
    def test_weights_sum_positive(self):
        preds = pd.Series({"A": 1, "B": 2, "C": 3})
        strategy = RankWeightedStrategy()
        weights = strategy.generate_weights(preds)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)


class TestLongShortStrategy:
    def test_long_short_sum_zero(self):
        preds = pd.Series({"A": 0.1, "B": 0.5, "C": 0.3, "D": 0.05, "E": 0.05})
        strategy = LongShortStrategy(top_pct=0.2, bottom_pct=0.2)
        weights = strategy.generate_weights(preds)
        assert weights.sum() == pytest.approx(0.0, abs=1e-6)
        assert weights["B"] > 0
        assert weights["D"] < 0
