import os
import tempfile
import pandas as pd
import pytest
from quant_trading.data.loader import CSVDataSource, DataLoader, AKShareDataSource


class TestCSVDataSource:
    def test_load_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=5),
                "open": [10, 11, 12, 13, 14],
                "high": [11, 12, 13, 14, 15],
                "low": [9, 10, 11, 12, 13],
                "close": [10.5, 11.5, 12.5, 13.5, 14.5],
                "volume": [1000, 2000, 3000, 4000, 5000],
            })
            path = os.path.join(tmpdir, "000001.csv")
            df.to_csv(path, index=False)

            src = CSVDataSource(tmpdir)
            loaded = src.load("000001", "2020-01-01", "2020-01-05")
            assert len(loaded) == 5
            assert "close" in loaded.columns

    def test_load_missing_csv(self):
        src = CSVDataSource("/tmp/nonexistent")
        with pytest.raises(FileNotFoundError):
            src.load("000001", "2020-01-01", "2020-01-05")


class TestDataLoaderCache:
    def test_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=3),
                "open": [10, 11, 12],
                "high": [11, 12, 13],
                "low": [9, 10, 11],
                "close": [10.5, 11.5, 12.5],
                "volume": [1000, 2000, 3000],
            })
            df.to_csv(os.path.join(tmpdir, "000001.csv"), index=False)
            loader = DataLoader(CSVDataSource(tmpdir))
            d1 = loader.load("000001", "2020-01-01", "2020-01-05")
            d2 = loader.load("000001", "2020-01-01", "2020-01-05")
            assert d1 is d2  # same object from cache

    def test_load_multi(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for sym in ["A", "B"]:
                df = pd.DataFrame({
                    "date": pd.date_range("2020-01-01", periods=3),
                    "open": [10, 11, 12],
                    "high": [11, 12, 13],
                    "low": [9, 10, 11],
                    "close": [10, 11, 12],
                    "volume": [100, 200, 300],
                })
                df.to_csv(os.path.join(tmpdir, f"{sym}.csv"), index=False)
            loader = DataLoader(CSVDataSource(tmpdir))
            result = loader.load_multi(["A", "B"], "2020-01-01", "2020-01-05")
            assert set(result.keys()) == {"A", "B"}
            assert len(result["A"]) == 3
