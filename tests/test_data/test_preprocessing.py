import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    calculate_ma,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    rolling_normalize,
    time_series_split,
)


class TestCalculateMA:
    def test_ma_basic(self):
        prices = pd.Series([1, 2, 3, 4, 5])
        ma = calculate_ma(prices, window=3, shift=False)
        assert pd.isna(ma.iloc[0])
        assert pd.isna(ma.iloc[1])
        assert ma.iloc[2] == pytest.approx(2.0)

    def test_ma_shift_prevents_lookahead(self):
        prices = pd.Series([1, 2, 3, 4, 5])
        ma = calculate_ma(prices, window=3, shift=True)
        assert pd.isna(ma.iloc[2])
        assert ma.iloc[3] == pytest.approx(2.0)


class TestCalculateRSI:
    def test_rsi_range(self):
        prices = pd.Series([10, 11, 12, 11, 10, 9, 10, 11, 12, 13])
        rsi = calculate_rsi(prices, window=5, shift=False)
        valid = rsi.dropna()
        assert ((valid >= 0) & (valid <= 100)).all()


class TestRollingNormalize:
    def test_rolling_normalize_no_lookahead(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        norm = rolling_normalize(df, window=5)
        assert norm["a"].iloc[:5].isna().all()
        assert not norm["a"].iloc[5:].isna().any()


class TestTimeSeriesSplit:
    def test_split_by_date(self):
        dates = pd.date_range("2020-01-01", periods=10)
        df = pd.DataFrame({"value": range(10)}, index=dates)
        train, val, test = time_series_split(df, train_end="2020-01-05", val_end="2020-01-07")
        assert len(train) == 5
        assert len(val) == 2
        assert len(test) == 3
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()
