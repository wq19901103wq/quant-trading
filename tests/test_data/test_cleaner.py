import pandas as pd
import pytest
from quant_trading.data.cleaner import DataCleaner


class TestDataCleaner:
    def test_remove_zero_volume(self):
        df = pd.DataFrame({
            "open": [10, 11, 12],
            "high": [11, 12, 13],
            "low": [9, 10, 11],
            "close": [10.5, 11.5, 12.5],
            "volume": [1000, 0, 2000],
        }, index=pd.date_range("2020-01-01", periods=3))
        cleaner = DataCleaner()
        cleaned = cleaner.clean(df)
        assert len(cleaned) == 2
        assert cleaner.get_report()["removed"] == 1

    def test_remove_negative_price(self):
        df = pd.DataFrame({
            "open": [10, -1, 12],
            "high": [11, 12, 13],
            "low": [9, 10, 11],
            "close": [10.5, 11.5, 12.5],
            "volume": [1000, 1000, 2000],
        }, index=pd.date_range("2020-01-01", periods=3))
        cleaner = DataCleaner()
        cleaned = cleaner.clean(df)
        assert len(cleaned) == 2

    def test_remove_price_relation_anomaly(self):
        df = pd.DataFrame({
            "open": [10, 11, 12],
            "high": [9, 12, 13],
            "low": [9, 10, 11],
            "close": [10.5, 11.5, 12.5],
            "volume": [1000, 1000, 2000],
        }, index=pd.date_range("2020-01-01", periods=3))
        cleaner = DataCleaner()
        cleaned = cleaner.clean(df)
        assert len(cleaned) == 2
