"""数据加载模块：AKShare + 本地 CSV 两种源."""
import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataSource(ABC):
    @abstractmethod
    def load(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass


class CSVDataSource(DataSource):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df.loc[start_date:end_date]


class AKShareDataSource(DataSource):
    """AKShare 数据源（默认新浪接口，支持东方财富备用）."""

    def __init__(self, adjust: str = "qfq", provider: str = "sina"):
        self.adjust = adjust
        self.provider = provider
        self._ak = None

    def _get_ak(self):
        if self._ak is None:
            try:
                import akshare as ak
                self._ak = ak
            except ImportError:
                raise ImportError("akshare not installed")
        return self._ak

    @staticmethod
    def _to_exchange_symbol(symbol: str) -> str:
        """将 000001 / 600036 转换为 sz000001 / sh600036."""
        if symbol.startswith(("sh", "sz", "bj")):
            return symbol
        code = int(symbol)
        if code >= 600000 or (code >= 300000 and code < 400000) or (code >= 430000 and code < 500000):
            return f"sh{symbol}"
        return f"sz{symbol}"

    def load(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        ak = self._get_ak()
        logger.info("Loading %s from AKShare (%s) [%s ~ %s]", symbol, self.provider, start_date, end_date)

        if self.provider == "sina":
            ex_symbol = self._to_exchange_symbol(symbol)
            df = ak.stock_zh_a_daily(symbol=ex_symbol, start_date=start_date, end_date=end_date, adjust=self.adjust)
            # 新浪返回列已经是英文: date, open, high, low, close, volume, amount, outstanding_share, turnover
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            # 只保留核心列
            keep = ["open", "high", "low", "close", "volume", "amount"]
            df = df[[c for c in keep if c in df.columns]]

        elif self.provider == "em":
            # 东方财富（可能被限流）
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust=self.adjust,
            )
            col_map = {
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
            }
            rename = {c: col_map[c] for c in df.columns if c in col_map}
            df = df.rename(columns=rename)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        if "symbol" not in df.columns:
            df["symbol"] = symbol
        return df


class CachedAKShareDataSource(DataSource):
    """带本地 CSV 缓存的 AKShare 数据源."""

    def __init__(self, cache_dir: str = "./data/cache", adjust: str = "qfq", provider: str = "sina", throttle_seconds: float = 1.0):
        self.cache_dir = cache_dir
        self.adjust = adjust
        self.provider = provider
        self.throttle_seconds = throttle_seconds
        self._ak_source = AKShareDataSource(adjust=adjust, provider=provider)
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, symbol: str, start_date: str, end_date: str) -> str:
        return os.path.join(self.cache_dir, f"{symbol}_{start_date}_{end_date}_{self.adjust}.csv")

    def load(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_path = self._cache_path(symbol, start_date, end_date)
        if os.path.exists(cache_path):
            logger.info("Cache hit for %s [%s ~ %s]", symbol, start_date, end_date)
            df = pd.read_csv(cache_path)
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date").sort_index()

        df = self._ak_source.load(symbol, start_date, end_date)
        df.reset_index().to_csv(cache_path, index=False)
        logger.info("Saved cache to %s", cache_path)
        time.sleep(self.throttle_seconds)
        return df

    def clear_cache(self):
        """清空本地缓存."""
        for f in os.listdir(self.cache_dir):
            if f.endswith(".csv"):
                os.remove(os.path.join(self.cache_dir, f))
        logger.info("Cache cleared")


class DataLoader:
    def __init__(self, source: DataSource):
        self.source = source
        self._cache: Dict[str, pd.DataFrame] = {}

    def load(self, symbol: str, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        df = self.source.load(symbol, start_date, end_date)
        if use_cache:
            self._cache[cache_key] = df
        return df

    def load_multi(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        return {s: self.load(s, start_date, end_date) for s in symbols}
