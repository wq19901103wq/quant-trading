"""时序交叉验证."""
from typing import List, Tuple
import pandas as pd
from src.models.base import Model


class TimeSeriesCV:
    def __init__(self, n_splits: int = 5, train_size: int = None):
        self.n_splits = n_splits
        self.train_size = train_size

    def split(self, X: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        splits = []
        for i in range(self.n_splits):
            test_start = (i + 1) * fold_size
            test_end = (i + 2) * fold_size if i < self.n_splits - 1 else n
            train_end = test_start
            if self.train_size:
                train_start = max(0, train_end - self.train_size)
            else:
                train_start = 0
            train_idx = X.index[train_start:train_end]
            test_idx = X.index[test_start:test_end]
            splits.append((train_idx, test_idx))
        return splits

    def cross_val_score(self, model: Model, X: pd.DataFrame, y: pd.Series, metric_fn) -> List[float]:
        scores = []
        for train_idx, test_idx in self.split(X):
            model_clone = model.__class__(model.get_params())
            model_clone.fit(X.loc[train_idx], y.loc[train_idx])
            preds = model_clone.predict(X.loc[test_idx])
            score = metric_fn(y.loc[test_idx], preds)
            scores.append(score)
        return scores
