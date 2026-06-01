"""价格因子."""
import pandas as pd
import numpy as np
from quant_trading.factors.base import Factor
from quant_trading.factors.registry import FactorRegistry


@FactorRegistry.register
class ReturnFactor(Factor):
    name = "return_1d"
    dependencies = ["close"]

    def __init__(self, period: int = 1):
        self.period = period
        self.name = f"return_{period}d"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change(self.period).shift(1)


@FactorRegistry.register
class VolatilityFactor(Factor):
    name = "volatility_20d"
    dependencies = ["close"]

    def __init__(self, window: int = 20):
        self.window = window
        self.name = f"volatility_{window}d"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change().rolling(self.window).std().shift(1)


@FactorRegistry.register
class HighLowRatioFactor(Factor):
    name = "high_low_ratio"
    dependencies = ["high", "low"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return (df["high"] / df["low"] - 1).shift(1)
