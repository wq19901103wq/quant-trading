import pandas as pd
import pytest
from src.factors.technical import MovingAverageFactor, RSIFactor, MACDFactor, BollingerUpperFactor


class TestMovingAverageFactor:
    def test_ma_shifted(self):
        df = pd.DataFrame({"close": [10.0] * 30})
        f = MovingAverageFactor(window=5)
        result = f.compute(df)
        assert pd.isna(result.iloc[4])
        assert result.iloc[5] == 10.0


class TestRSIFactor:
    def test_rsi_range(self):
        df = pd.DataFrame({"close": [10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15] * 3})
        f = RSIFactor(window=5)
        result = f.compute(df)
        valid = result.dropna()
        assert ((valid >= 0) & (valid <= 100)).all()
