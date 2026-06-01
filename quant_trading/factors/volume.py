"""成交量因子."""
import pandas as pd
import numpy as np
from quant_trading.factors.base import Factor
from quant_trading.factors.registry import FactorRegistry


@FactorRegistry.register
class VolumeMAFactor(Factor):
    name = "volume_ma_20"
    dependencies = ["volume"]

    def __init__(self, window: int = 20):
        self.window = window
        self.name = f"volume_ma_{window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["volume"].rolling(self.window).mean().shift(1)


@FactorRegistry.register
class OBVFactor(Factor):
    name = "obv"
    dependencies = ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        sign = np.sign(df["close"].diff())
        obv = (sign * df["volume"]).cumsum()
        return obv.shift(1)
