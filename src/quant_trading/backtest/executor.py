"""交易执行模拟器."""
from typing import Dict, Optional
import pandas as pd
import numpy as np


class Executor:
    def __init__(self, commission_rate: float = 0.001, slippage: float = 0.0, min_commission: float = 5.0):
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.min_commission = min_commission

    def execute(self, target_weights: pd.Series, current_weights: pd.Series, prices: pd.Series, capital: float) -> Dict:
        trade_weights = target_weights - current_weights
        trade_value = trade_weights.abs() * capital
        commission = trade_value * self.commission_rate
        commission = commission.clip(lower=self.min_commission).where(trade_value > 0, 0)
        slippage_cost = trade_value * self.slippage
        total_cost = commission.sum() + slippage_cost.sum()
        executed_weights = target_weights.copy()
        return {
            "target_weights": target_weights,
            "executed_weights": executed_weights,
            "trade_weights": trade_weights,
            "commission": commission.sum(),
            "slippage_cost": slippage_cost.sum(),
            "total_cost": total_cost,
        }
