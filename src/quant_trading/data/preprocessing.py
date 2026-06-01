"""数据预处理模块：底层数学工具（无状态，不产因子）."""
import numpy as np
import pandas as pd
from typing import Tuple


def calculate_ma(prices: pd.Series, window: int = 20, shift: bool = True) -> pd.Series:
    ma = prices.rolling(window=window, min_periods=window).mean()
    return ma.shift(1) if shift else ma


def calculate_ema(prices: pd.Series, span: int = 20, shift: bool = True) -> pd.Series:
    ema = prices.ewm(span=span, adjust=False).mean()
    return ema.shift(1) if shift else ema


def calculate_rsi(prices: pd.Series, window: int = 14, shift: bool = True) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.shift(1) if shift else rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, shift: bool = True) -> pd.DataFrame:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    if shift:
        macd_line = macd_line.shift(1)
        signal_line = signal_line.shift(1)
        histogram = histogram.shift(1)
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0, shift: bool = True) -> pd.DataFrame:
    middle = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = middle + std * num_std
    lower = middle - std * num_std
    if shift:
        upper = upper.shift(1)
        middle = middle.shift(1)
        lower = lower.shift(1)
    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})


def rolling_normalize(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for col in df.columns:
        rolling_mean = df[col].rolling(window=window, min_periods=window).mean().shift(1)
        rolling_std = df[col].rolling(window=window, min_periods=window).std().shift(1)
        rolling_std = rolling_std.replace(0, np.nan)
        result[col] = (df[col] - rolling_mean) / rolling_std
    return result


def time_series_split(df: pd.DataFrame, train_end: str, val_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df.index <= train_end]
    val_df = df[(df.index > train_end) & (df.index <= val_end)]
    test_df = df[df.index > val_end]
    return train_df, val_df, test_df
