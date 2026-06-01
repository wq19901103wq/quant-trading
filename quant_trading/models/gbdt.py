"""LightGBM GBDT 模型封装."""
from typing import Any, Dict
import pandas as pd
import numpy as np
import lightgbm as lgb
from quant_trading.models.base import Model


class LightGBMModel(Model):
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }
        self.model: lgb.Booster = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, early_stopping_rounds=None) -> None:
        train_data = lgb.Dataset(X, label=y)
        valid_sets = [train_data]
        if eval_set:
            valid_sets.append(lgb.Dataset(eval_set[0], label=eval_set[1]))
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=100,
            valid_sets=valid_sets,
            callbacks=[lgb.log_evaluation(period=0)],
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        return self.model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        importance = self.model.feature_importance(importance_type="gain")
        return pd.Series(importance, index=self.model.feature_name())
