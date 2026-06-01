import pandas as pd
import pytest
from src.factors.price import ReturnFactor, VolatilityFactor, HighLowRatioFactor


class TestReturnFactor:
    def test_return_shifted(self):
        df = pd.DataFrame({"close": [10, 11, 12, 13, 14]})
        f = ReturnFactor(period=1)
        result = f.compute(df)
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(0.1)  # (11-10)/10


class TestVolatilityFactor:
    def test_volatility_non_negative(self):
        df = pd.DataFrame({"close": [10, 11, 12, 11, 10, 9, 10, 11, 12, 13] * 3})
        f = VolatilityFactor(window=5)
        result = f.compute(df)
        assert (result.dropna() >= 0).all()
