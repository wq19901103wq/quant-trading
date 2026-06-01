"""数据清洗模块."""
import pandas as pd
from typing import Dict


class DataCleaner:
    def __init__(
        self,
        min_price: float = 0.01,
        max_daily_change: float = 0.21,
        min_volume: float = 1.0,
        max_consecutive_limit: int = 5,
    ):
        self.min_price = min_price
        self.max_daily_change = max_daily_change
        self.min_volume = min_volume
        self.max_consecutive_limit = max_consecutive_limit
        self.report = {}

    def clean(self, df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        original_len = len(df)
        removed = {"total": 0, "steps": []}
        df = df.copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        before = len(df)
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
        n = before - len(df)
        if n > 0:
            removed["steps"].append(f"non-positive price: {n}")
            removed["total"] += n

        before = len(df)
        df = df[df["volume"] >= self.min_volume]
        n = before - len(df)
        if n > 0:
            removed["steps"].append(f"zero/low volume: {n}")
            removed["total"] += n

        before = len(df)
        valid = (
            (df["high"] >= df["low"])
            & (df["high"] >= df[["open", "close"]].max(axis=1))
            & (df["low"] <= df[["open", "close"]].min(axis=1))
        )
        df = df[valid]
        n = before - len(df)
        if n > 0:
            removed["steps"].append(f"price relation anomaly: {n}")
            removed["total"] += n

        before = len(df)
        returns = df["close"].pct_change().abs()
        df = df[(returns <= self.max_daily_change) | returns.isna()]
        n = before - len(df)
        if n > 0:
            removed["steps"].append(f"extreme return: {n}")
            removed["total"] += n

        before = len(df)
        price_change = df["close"].diff().abs()
        consecutive_unchanged = price_change.rolling(self.max_consecutive_limit).sum() == 0
        df = df[~consecutive_unchanged.fillna(False)]
        n = before - len(df)
        if n > 0:
            removed["steps"].append(f"consecutive unchanged: {n}")
            removed["total"] += n

        before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        n = before - len(df)
        if n > 0:
            removed["steps"].append(f"duplicate date: {n}")
            removed["total"] += n

        self.report = {
            "symbol": symbol,
            "original": original_len,
            "final": len(df),
            "removed": removed["total"],
            "retention": len(df) / original_len if original_len > 0 else 0,
            "details": removed["steps"],
        }
        return df

    def get_report(self) -> Dict:
        return self.report
