"""技术指标因子."""
import pandas as pd
from src.factors.base import Factor
from src.data.preprocessing import calculate_ma, calculate_rsi, calculate_macd, calculate_bollinger_bands
from src.factors.registry import FactorRegistry


@FactorRegistry.register
class MovingAverageFactor(Factor):
    name = "ma_20"
    dependencies = ["close"]

    def __init__(self, window: int = 20):
        self.window = window
        self.name = f"ma_{window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return calculate_ma(df["close"], window=self.window, shift=True)


@FactorRegistry.register
class RSIFactor(Factor):
    name = "rsi_14"
    dependencies = ["close"]

    def __init__(self, window: int = 14):
        self.window = window
        self.name = f"rsi_{window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return calculate_rsi(df["close"], window=self.window, shift=True)


@FactorRegistry.register
class MACDFactor(Factor):
    name = "macd"
    dependencies = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        macd_df = calculate_macd(df["close"], shift=True)
        return macd_df["histogram"]


@FactorRegistry.register
class BollingerUpperFactor(Factor):
    name = "bb_upper"
    dependencies = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        bb = calculate_bollinger_bands(df["close"], shift=True)
        return bb["upper"]
