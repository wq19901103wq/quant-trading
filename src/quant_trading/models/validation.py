"""模型验证工具."""
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def ic_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return pd.Series(y_true).corr(pd.Series(y_pred))


def rank_ic_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
