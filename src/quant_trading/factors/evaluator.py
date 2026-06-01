"""因子评估器：IC、 turnover、decay 等."""
from typing import Dict
import pandas as pd
import numpy as np


class FactorEvaluator:
    def evaluate(self, factor: pd.Series, forward_return: pd.Series) -> Dict:
        aligned = pd.concat([factor, forward_return], axis=1).dropna()
        if len(aligned) < 2:
            return {"ic": np.nan, "rank_ic": np.nan, "ic_ir": np.nan}
        ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        rank_ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
        ic_ir = ic / aligned.iloc[:, 0].std() if aligned.iloc[:, 0].std() != 0 else np.nan
        return {"ic": ic, "rank_ic": rank_ic, "ic_ir": ic_ir}

    def evaluate_group(self, df: pd.DataFrame, factor_col: str, return_col: str) -> Dict:
        group_result = {}
        for date, group in df.groupby(df.index):
            ic = group[factor_col].corr(group[return_col])
            group_result[date] = ic
        ic_series = pd.Series(group_result)
        return {
            "mean_ic": ic_series.mean(),
            "ic_std": ic_series.std(),
            "ir": ic_series.mean() / ic_series.std() if ic_series.std() != 0 else np.nan,
        }

    def turnover(self, factor: pd.Series, by_date: bool = False) -> float:
        if by_date:
            dates = factor.index.get_level_values(0).unique() if isinstance(factor.index, pd.MultiIndex) else factor.index
            # simplified: assume single series
            return np.nan
        diff = factor.diff().abs()
        total = factor.abs().sum()
        if total == 0:
            return 0.0
        return diff.sum() / total
