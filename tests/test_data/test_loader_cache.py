import os
import tempfile
import pandas as pd
import pytest
from src.data.loader import CachedAKShareDataSource, CSVDataSource


class TestCachedAKShareDataSource:
    def test_cache_miss_then_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 先准备一个模拟的 CSV 缓存
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            df = pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=5),
                "open": [10, 11, 12, 13, 14],
                "high": [11, 12, 13, 14, 15],
                "low": [9, 10, 11, 12, 13],
                "close": [10.5, 11.5, 12.5, 13.5, 14.5],
                "volume": [1000, 2000, 3000, 4000, 5000],
                "symbol": ["000001"] * 5,
            })
            cache_path = os.path.join(cache_dir, "000001_2020-01-01_2020-01-31_qfq.csv")
            df.to_csv(cache_path, index=False)

            source = CachedAKShareDataSource(cache_dir=cache_dir)
            loaded = source.load("000001", "2020-01-01", "2020-01-31")
            assert len(loaded) == 5
            assert "close" in loaded.columns
            assert loaded.index.name == "date"

    def test_clear_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, "dummy.csv"), "w") as f:
                f.write("a,b\n1,2\n")
            source = CachedAKShareDataSource(cache_dir=cache_dir)
            source.clear_cache()
            assert len(os.listdir(cache_dir)) == 0
