import pandas as pd
import pytest
from src.factors.base import Factor


class DummyFactor(Factor):
    name = "dummy"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] * 2


class TestFactor:
    def test_compute(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        f = DummyFactor()
        result = f.compute(df)
        assert result.iloc[0] == 2
