import os
import tempfile
import pandas as pd
import pytest
from quant_trading.data.loader import CSVDataSource, DataLoader
from quant_trading.data.cleaner import DataCleaner
from quant_trading.data.handler import DataHandler


class TestDataHandler:
    def _make_data(self, tmpdir: str, symbol: str):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10),
            "open": list(range(10, 20)),
            "high": list(range(11, 21)),
            "low": list(range(9, 19)),
            "close": list(range(10, 20)),
            "volume": [100] * 10,
        })
        df.to_csv(os.path.join(tmpdir, f"{symbol}.csv"), index=False)

    def test_handler_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, "000001")
            loader = DataLoader(CSVDataSource(tmpdir))
            handler = DataHandler(
                data_loader=loader,
                symbols=["000001"],
                start_date="2020-01-01",
                end_date="2020-01-10",
                features=["close", "volume"],
                label="close",
            )
            data = handler.get_data()
            assert len(data) == 10
            assert set(handler.get_feature_cols()) == {"close", "volume"}

    def test_handler_missing_feature_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, "000001")
            loader = DataLoader(CSVDataSource(tmpdir))
            handler = DataHandler(
                data_loader=loader,
                symbols=["000001"],
                start_date="2020-01-01",
                end_date="2020-01-10",
                features=["nonexistent"],
                label="close",
            )
            with pytest.raises(ValueError, match="Missing columns"):
                handler.get_data()

    def test_handler_na_drop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=5),
                "open": [10, 11, 12, 13, 14],
                "high": [11, 12, 13, 14, 15],
                "low": [9, 10, 11, 12, 13],
                "close": [10, None, 12, None, 14],
                "volume": [100, 200, 300, 400, 500],
            })
            df.to_csv(os.path.join(tmpdir, "000001.csv"), index=False)
            loader = DataLoader(CSVDataSource(tmpdir))
            handler = DataHandler(
                data_loader=loader,
                symbols=["000001"],
                start_date="2020-01-01",
                end_date="2020-01-10",
                features=["close", "volume"],
                label="close",
                na_method="drop",
            )
            data = handler.get_data()
            assert len(data) == 3
