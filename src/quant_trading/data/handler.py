"""DataHandler：将 loader + cleaner + factors 组装为统一入口."""
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import logging

from quant_trading.data.loader import DataLoader
from quant_trading.data.cleaner import DataCleaner

logger = logging.getLogger(__name__)


class DataHandler:
    def __init__(
        self,
        data_loader: DataLoader,
        symbols: List[str],
        start_date: str,
        end_date: str,
        features: List[str],
        label: str,
        cleaner: Optional[DataCleaner] = None,
        pre_processor: Optional[callable] = None,
        na_method: str = "drop",
    ):
        self.data_loader = data_loader
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.features = features
        self.label = label
        self.cleaner = cleaner or DataCleaner()
        self.pre_processor = pre_processor
        self.na_method = na_method
        self._data: Optional[pd.DataFrame] = None
        self._feature_cols: List[str] = []
        self._label_col: str = ""

    def load_data(self) -> pd.DataFrame:
        all_frames = []
        for sym in self.symbols:
            try:
                raw = self.data_loader.load(sym, self.start_date, self.end_date)
                cleaned = self.cleaner.clean(raw, symbol=sym)
                cleaned["symbol"] = sym
                all_frames.append(cleaned)
            except FileNotFoundError:
                logger.warning("Data not found for %s, skipping", sym)
        if not all_frames:
            return pd.DataFrame()
        df = pd.concat(all_frames)
        if self.pre_processor:
            df = self.pre_processor(df)
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing = [c for c in self.features + [self.label] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self._feature_cols = self.features
        self._label_col = self.label
        if self.na_method == "drop":
            df = df.dropna(subset=self._feature_cols + [self._label_col])
        elif self.na_method == "ffill":
            df[self._feature_cols + [self._label_col]] = df[self._feature_cols + [self._label_col]].fillna(method="ffill")
        elif self.na_method == "zero":
            df[self._feature_cols + [self._label_col]] = df[self._feature_cols + [self._label_col]].fillna(0)
        return df

    def get_data(self) -> pd.DataFrame:
        if self._data is None:
            df = self.load_data()
            self._data = self.prepare_features(df)
        return self._data

    def get_feature_matrix(self) -> pd.DataFrame:
        df = self.get_data()
        return df[self._feature_cols]

    def get_label_series(self) -> pd.Series:
        df = self.get_data()
        return df[self._label_col]

    def get_feature_cols(self) -> List[str]:
        return self._feature_cols.copy()

    def get_label_col(self) -> str:
        return self._label_col
