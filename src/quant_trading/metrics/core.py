"""回测核心指标."""
from typing import Dict
import pandas as pd
import numpy as np


def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0.0
    volatility = returns.std() * np.sqrt(252)
    sharpe = annualized_return / volatility if volatility != 0 else 0.0
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.0
    sortino = annualized_return / downside_std if downside_std != 0 else 0.0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
    }
