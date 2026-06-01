import os
import tempfile
import pandas as pd
import numpy as np
from quant_trading.data.loader import CSVDataSource, DataLoader
from quant_trading.data.cleaner import DataCleaner
from quant_trading.data.handler import DataHandler
from quant_trading.factors.registry import FactorRegistry
from quant_trading.factors.technical import MovingAverageFactor
from quant_trading.models.gbdt import LightGBMModel
from quant_trading.models.cross_validation import TimeSeriesCV
from quant_trading.portfolio.strategy import TopKStrategy
from quant_trading.backtest.executor import Executor
from quant_trading.backtest.engine import BacktestEngine
from quant_trading.experiments.recorder import Recorder
from quant_trading.pipeline.rolling import RollingBacktest


class TestEndToEnd:
    def _make_data(self, tmpdir, symbol):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame({
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1000, 10000, 100),
            "ma_5": pd.Series(close).rolling(5).mean().shift(1).values,
            "return_1d": pd.Series(close).pct_change().shift(-1).values,
        })
        df.to_csv(os.path.join(tmpdir, f"{symbol}.csv"), index=False)

    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, "000001")

            # 1. Load data
            loader = DataLoader(CSVDataSource(tmpdir))
            handler = DataHandler(
                data_loader=loader,
                symbols=["000001"],
                start_date="2020-01-01",
                end_date="2020-12-31",
                features=["ma_5"],
                label="return_1d",
            )
            df = handler.get_data()
            assert len(df) > 0
            assert "ma_5" in df.columns

            # 2. Train model
            X = handler.get_feature_matrix()
            y = handler.get_label_series()
            model = LightGBMModel(params={"objective": "regression", "verbose": -1, "seed": 42})
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == len(X)

            # 3. Portfolio + backtest
            predictions_df = pd.DataFrame({"000001": preds}, index=X.index)
            prices_df = pd.DataFrame({"000001": df["close"].values}, index=df.index)
            engine = BacktestEngine(
                strategy=TopKStrategy(k=1),
                executor=Executor(commission_rate=0.0),
                initial_capital=1_000_000,
            )
            result = engine.run(predictions_df, prices_df)
            assert "metrics" in result
            assert result["metrics"]["total_return"] is not None

            # 4. Recorder
            recorder = Recorder(tmpdir, "integration")
            recorder.log_params({"symbol": "000001"})
            recorder.log_metrics(result["metrics"])
            assert os.path.exists(os.path.join(recorder.get_run_dir(), "params.json"))
            assert os.path.exists(os.path.join(recorder.get_run_dir(), "metrics.json"))
