"""滚动回测流水线."""
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from quant_trading.data.handler import DataHandler
from quant_trading.models.base import Model
from quant_trading.portfolio.strategy import PortfolioStrategy
from quant_trading.backtest.engine import BacktestEngine
from quant_trading.backtest.executor import Executor
from quant_trading.metrics.core import calculate_metrics
from quant_trading.experiments.recorder import Recorder
from quant_trading.models.validation import evaluate_regression


class RollingBacktest:
    def __init__(
        self,
        data_handler: DataHandler,
        model: Model,
        strategy: PortfolioStrategy,
        executor: Executor,
        train_window: int = 252,
        test_window: int = 63,
        step: int = 63,
        label_col: Optional[str] = None,
        recorder: Optional[Recorder] = None,
    ):
        self.data_handler = data_handler
        self.model = model
        self.strategy = strategy
        self.executor = executor
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.label_col = label_col
        self.recorder = recorder

    def run(self) -> Dict:
        df = self.data_handler.get_data()
        if df.empty:
            raise ValueError("No data loaded")
        features = self.data_handler.get_feature_cols()
        label_col = self.label_col or self.data_handler.get_label_col()
        dates = df.index.unique()
        n = len(dates)
        all_predictions = []
        all_returns = []
        fold_results = []

        for start in range(self.train_window, n - self.test_window, self.step):
            train_start = start - self.train_window
            train_end = start
            test_end = min(start + self.test_window, n)
            train_dates = dates[train_start:train_end]
            test_dates = dates[train_end:test_end]

            train_df = df.loc[df.index.isin(train_dates)]
            test_df = df.loc[df.index.isin(test_dates)]

            X_train = train_df[features]
            y_train = train_df[label_col]
            X_test = test_df[features]
            y_test = test_df[label_col]

            model_clone = self.model.__class__(self.model.get_params())
            model_clone.fit(X_train, y_train)
            preds = model_clone.predict(X_test)

            pred_series = pd.Series(preds, index=test_df.index)
            all_predictions.append(pred_series)

            if "close" in test_df.columns:
                # simplistic single-symbol assumption for integration test
                returns = test_df["close"].pct_change().shift(-1).reindex(test_df.index)
                all_returns.append(returns)

            fold_results.append({
                "train_start": str(train_dates[0]),
                "train_end": str(train_dates[-1]),
                "test_start": str(test_dates[0]),
                "test_end": str(test_dates[-1]),
                "train_size": len(train_df),
                "test_size": len(test_df),
            })

        if self.recorder:
            self.recorder.log_params({
                "train_window": self.train_window,
                "test_window": self.test_window,
                "step": self.step,
            })
            if all_predictions:
                combined_preds = pd.concat(all_predictions)
                self.recorder.log_artifact("predictions.csv", combined_preds)
            self.recorder.log_metrics({"num_folds": len(fold_results)})

        return {
            "folds": fold_results,
            "predictions": pd.concat(all_predictions) if all_predictions else pd.Series(),
            "metrics": {"num_folds": len(fold_results)},
        }
