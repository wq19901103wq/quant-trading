"""数据加载模块：AKShare + 本地 CSV 两种源."""
import os
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
    def __init__(self, adjust: str = "qfq"):
        self.adjust = adjust
        self._ak = None

    def _get_ak(self):
        if self._ak is None:
            try:
                import akshare as ak
                self._ak = ak
            except ImportError:
                raise ImportError("akshare not installed")
        return self._ak

    def load(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        ak = self._get_ak()
        logger.info("Loading %s from AKShare [%s ~ %s]", symbol, start_date, end_date)
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date.replace("-", ""), end_date=end_date.replace("-", ""), adjust=self.adjust)
        df.columns = [c.lower().strip() for c in df.columns]
        if "日期" in df.columns:
            df = df.rename(columns={"日期": "date"})
        elif "date" not in df.columns:
            for c in df.columns:
                if "date" in c or "日期" in c:
                    df = df.rename(columns={c: "date"})
                    break
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        col_map = {
            "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount",
            "open": "open", "close": "close", "high": "high", "low": "low",
            "volume": "volume", "amount": "amount",
        }
        rename = {}
        for c in df.columns:
            if c in col_map:
                rename[c] = col_map[c]
        if rename:
            df = df.rename(columns=rename)
        return df


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
