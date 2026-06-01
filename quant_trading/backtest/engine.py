"""回测引擎."""
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from quant_trading.portfolio.strategy import PortfolioStrategy
from quant_trading.backtest.executor import Executor
from quant_trading.metrics.core import calculate_metrics


class BacktestEngine:
    def __init__(
        self,
        strategy: PortfolioStrategy,
        executor: Executor,
        initial_capital: float = 1_000_000,
    ):
        self.strategy = strategy
        self.executor = executor
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.history: List[Dict] = []
        self.daily_returns: List[float] = []

    def run(
        self,
        predictions_df: pd.DataFrame,
        prices_df: pd.DataFrame,
    ) -> Dict:
        dates = predictions_df.index
        current_weights = pd.Series(0.0, index=predictions_df.columns)
        portfolio_values = []

        for date in dates:
            preds = predictions_df.loc[date]
            target_weights = self.strategy.generate_weights(preds, current_weights)
            prices = prices_df.loc[date] if date in prices_df.index else pd.Series(np.nan, index=predictions_df.columns)
            execution = self.executor.execute(target_weights, current_weights, prices, self.capital)
            current_weights = execution["executed_weights"]

            if date in prices_df.index:
                next_date_idx = prices_df.index.get_loc(date) + 1
                if next_date_idx < len(prices_df):
                    next_prices = prices_df.iloc[next_date_idx]
                    price_change = next_prices / prices - 1
                    valid = price_change.notna()
                    portfolio_return = (current_weights[valid] * price_change[valid]).sum()
                else:
                    portfolio_return = 0.0
            else:
                portfolio_return = 0.0

            portfolio_values.append(self.capital)
            self.capital *= (1 + portfolio_return)
            self.daily_returns.append(portfolio_return)
            self.history.append({
                "date": date,
                "target_weights": target_weights.to_dict(),
                "executed_weights": current_weights.to_dict(),
                "portfolio_return": portfolio_return,
                "portfolio_value": self.capital,
            })

        returns_series = pd.Series(self.daily_returns, index=dates)
        metrics = calculate_metrics(returns_series)
        return {
            "metrics": metrics,
            "returns": returns_series,
            "portfolio_values": pd.Series(portfolio_values, index=dates),
            "history": self.history,
        }
