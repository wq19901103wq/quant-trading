import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from quant_trading.pipeline.rolling import RollingBacktest
from quant_trading.data.loader import CSVDataSource, DataLoader
from quant_trading.data.handler import DataHandler
from quant_trading.models.gbdt import LightGBMModel
from quant_trading.portfolio.strategy import TopKStrategy
from quant_trading.backtest.executor import Executor
from quant_trading.experiments.recorder import Recorder


class TestRollingBacktest:
    def _make_csv(self, tmpdir, symbol, n=100):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=n)
        close = 100 + np.cumsum(np.random.randn(n))
        df = pd.DataFrame({
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "label": np.random.randn(n),
        })
        df.to_csv(os.path.join(tmpdir, f"{symbol}.csv"), index=False)

    def test_rolling_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_csv(tmpdir, "000001", n=200)
            loader = DataLoader(CSVDataSource(tmpdir))
            handler = DataHandler(
                data_loader=loader,
                symbols=["000001"],
                start_date="2020-01-01",
                end_date="2020-12-31",
                features=["feature1", "feature2"],
                label="label",
            )
            model = LightGBMModel(params={"objective": "regression", "verbose": -1, "seed": 42})
            rb = RollingBacktest(
                data_handler=handler,
                model=model,
                strategy=TopKStrategy(k=1),
                executor=Executor(commission_rate=0.0),
                train_window=50,
                test_window=20,
                step=20,
            )
            result = rb.run()
            assert "folds" in result
            assert len(result["folds"]) > 0
            assert len(result["predictions"]) > 0

    def test_rolling_with_recorder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_csv(tmpdir, "000001", n=200)
            loader = DataLoader(CSVDataSource(tmpdir))
            handler = DataHandler(
                data_loader=loader,
                symbols=["000001"],
                start_date="2020-01-01",
                end_date="2020-12-31",
                features=["feature1", "feature2"],
                label="label",
            )
            model = LightGBMModel(params={"objective": "regression", "verbose": -1, "seed": 42})
            recorder = Recorder(tmpdir, "rolling_test")
            rb = RollingBacktest(
                data_handler=handler,
                model=model,
                strategy=TopKStrategy(k=1),
                executor=Executor(commission_rate=0.0),
                train_window=50,
                test_window=20,
                step=20,
                recorder=recorder,
            )
            rb.run()
            assert os.path.exists(os.path.join(recorder.get_run_dir(), "params.json"))
            assert os.path.exists(os.path.join(recorder.get_run_dir(), "predictions.csv"))
