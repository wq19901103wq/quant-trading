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


def calculate_excess_metrics(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    """计算超额收益指标.
    
    Args:
        portfolio_returns: 组合日收益序列
        benchmark_returns: 基准日收益序列（需与组合对齐）
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return {
            "excess_return": np.nan,
            "annualized_excess_return": np.nan,
            "tracking_error": np.nan,
            "information_ratio": np.nan,
            "excess_max_drawdown": np.nan,
            "beta": np.nan,
        }
    
    port = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]
    excess = port - bench
    
    # 超额收益
    total_excess = (1 + excess).prod() - 1
    ann_excess = (1 + total_excess) ** (252 / len(excess)) - 1 if len(excess) > 0 else 0.0
    
    # 跟踪误差
    tracking_error = excess.std() * np.sqrt(252)
    
    # 信息比率
    ir = ann_excess / tracking_error if tracking_error != 0 else 0.0
    
    # 超额收益最大回撤
    cum_excess = (1 + excess).cumprod()
    running_max = cum_excess.cummax()
    excess_dd = (cum_excess - running_max) / running_max
    excess_max_dd = excess_dd.min()
    
    # Beta
    cov = np.cov(port, bench)[0, 1]
    var = np.var(bench)
    beta = cov / var if var != 0 else np.nan
    
    return {
        "excess_return": total_excess,
        "annualized_excess_return": ann_excess,
        "tracking_error": tracking_error,
        "information_ratio": ir,
        "excess_max_drawdown": excess_max_dd,
        "beta": beta,
    }
