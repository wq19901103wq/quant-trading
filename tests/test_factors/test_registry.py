import pytest
from quant_trading.factors.registry import FactorRegistry
from quant_trading.factors.base import Factor
import pandas as pd


class MyFactor(Factor):
    name = "my_factor"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"]


class TestFactorRegistry:
    def setup_method(self):
        FactorRegistry.clear()

    def test_register_and_get(self):
        FactorRegistry.register(MyFactor)
        cls = FactorRegistry.get("my_factor")
        assert cls is MyFactor

    def test_get_missing_raises(self):
        with pytest.raises(KeyError):
            FactorRegistry.get("missing")

    def test_list_factors(self):
        FactorRegistry.register(MyFactor)
        assert "my_factor" in FactorRegistry.list_factors()
