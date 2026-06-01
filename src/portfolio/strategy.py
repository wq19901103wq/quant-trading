"""组合策略：从模型预测生成目标权重."""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
import numpy as np


class PortfolioStrategy(ABC):
    @abstractmethod
    def generate_weights(self, predictions: pd.Series, current_holdings: Optional[pd.Series] = None) -> pd.Series:
        pass


class TopKStrategy(PortfolioStrategy):
    def __init__(self, k: int = 10, long_only: bool = True):
        self.k = k
        self.long_only = long_only

    def generate_weights(self, predictions: pd.Series, current_holdings: Optional[pd.Series] = None) -> pd.Series:
        top_k = predictions.nlargest(self.k)
        weights = pd.Series(0.0, index=predictions.index)
        weights.loc[top_k.index] = 1.0 / self.k
        return weights


class RankWeightedStrategy(PortfolioStrategy):
    def __init__(self, long_only: bool = True):
        self.long_only = long_only

    def generate_weights(self, predictions: pd.Series, current_holdings: Optional[pd.Series] = None) -> pd.Series:
        ranks = predictions.rank()
        weights = (ranks - ranks.min()) / (ranks.max() - ranks.min())
        if not self.long_only:
            weights = weights - 0.5
        weights = weights / weights.abs().sum()
        return weights.fillna(0)


class LongShortStrategy(PortfolioStrategy):
    def __init__(self, top_pct: float = 0.1, bottom_pct: float = 0.1):
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct

    def generate_weights(self, predictions: pd.Series, current_holdings: Optional[pd.Series] = None) -> pd.Series:
        n = len(predictions)
        top_n = max(1, int(n * self.top_pct))
        bottom_n = max(1, int(n * self.bottom_pct))
        top_idx = predictions.nlargest(top_n).index
        bottom_idx = predictions.nsmallest(bottom_n).index
        weights = pd.Series(0.0, index=predictions.index)
        weights.loc[top_idx] = 0.5 / top_n
        weights.loc[bottom_idx] = -0.5 / bottom_n
        return weights
