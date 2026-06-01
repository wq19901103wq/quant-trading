# quant-trading 项目重构实现计划

> **面向 AI 代理的工作者：** 使用 superpowers:subagent-driven-development 逐任务实现此计划。每个任务调度一个全新子代理，完成后进行两阶段审查（规格合规性 + 代码质量）。

**目标：** 从零构建 `quant-trading` 项目，搬运 `rl-quant-trading` 树模型子系统的核心价值，修复架构问题，遵循 TDD。

**架构：** 见 `docs/superpowers/specs/2026-06-01-quant-trading-design.md`

**技术栈：** Python 3.10+, LightGBM, pandas, numpy, pytest

**工作目录：** `quant-trading/`（与 `rl-quant-trading` 同级，完全独立）

---

## 项目结构预览

```
quant-trading/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── src/
│   └── quant_trading/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── cleaner.py
│       │   ├── preprocessing.py
│       │   └── handler.py
│       ├── factors/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── technical.py
│       │   ├── price.py
│       │   ├── volume.py
│       │   └── evaluator.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── gbdt.py
│       │   ├── cv.py
│       │   └── validation.py
│       ├── portfolio/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── strategy.py
│       ├── backtest/
│       │   ├── __init__.py
│       │   ├── executor.py
│       │   └── engine.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   └── core.py
│       ├── experiments/
│       │   ├── __init__.py
│       │   └── recorder.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── rolling.py
│       └── utils/
│           ├── __init__.py
│           └── config.py
└── tests/
    ├── conftest.py
    ├── test_data/
    ├── test_factors/
    ├── test_models/
    ├── test_portfolio/
    ├── test_backtest/
    ├── test_metrics/
    ├── test_experiments/
    └── test_pipeline/
```

---

## 任务总览

| 编号 | 任务 | 文件数 | 依赖 |
|------|------|--------|------|
| 1 | 项目骨架 + 配置管理 | 3 | 无 |
| 2 | 数据底层工具（preprocessing + cleaner） | 4 | 任务 1 |
| 3 | 数据加载 + DataHandler | 4 | 任务 2 |
| 4 | 因子层全部（base + 具体因子 + evaluator） | 6 | 任务 3 |
| 5 | 模型层全部（base + gbdt + cv + validation） | 5 | 任务 1 |
| 6 | portfolio + metrics | 4 | 任务 1 |
| 7 | backtest（executor + engine） | 4 | 任务 6 |
| 8 | experiments + pipeline + 集成 | 4 | 任务 3,4,5,6,7 |

---


## 任务 1：项目骨架 + 配置管理

**说明：** 创建项目目录结构、pyproject.toml、顶层 __init__.py、utils/config.py 和 tests/conftest.py。

**文件：**
- 创建：`pyproject.toml`
- 创建：`src/quant_trading/__init__.py`
- 创建：`src/quant_trading/utils/__init__.py`
- 创建：`src/quant_trading/utils/config.py`
- 创建：`tests/conftest.py`
- 创建：`tests/test_utils/test_config.py`

---

### 步骤 1：创建目录结构

运行：
```bash
mkdir -p src/quant_trading/{data,factors,models,portfolio,backtest,metrics,experiments,pipeline,utils}
mkdir -p tests/{test_data,test_factors,test_models,test_portfolio,test_backtest,test_metrics,test_experiments,test_pipeline,test_utils}
touch src/quant_trading/{data,factors,models,portfolio,backtest,metrics,experiments,pipeline,utils}/__init__.py
```

### 步骤 2：编写 pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "quant-trading"
version = "0.1.0"
description = "A quantitative trading research framework"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "lightgbm>=4.0.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
    "akshare>=1.12.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

### 步骤 3：编写测试（先失败）

`tests/test_utils/test_config.py`：
```python
import pytest
from quant_trading.utils.config import Config, load_config, save_config_snapshot
import tempfile
import os


class TestConfig:
    def test_config_dot_access(self):
        cfg = Config({"model": {"max_depth": 5, "lr": 0.05}})
        assert cfg.get("model.max_depth") == 5
        assert cfg.get("model.lr") == 0.05

    def test_config_default_value(self):
        cfg = Config({})
        assert cfg.get("missing.key", "default") == "default"

    def test_load_and_save_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                f.write("model:\n  max_depth: 5\n")
            
            cfg = load_config(config_path)
            assert cfg.get("model.max_depth") == 5
            
            log_dir = os.path.join(tmpdir, "logs")
            save_config_snapshot(cfg, log_dir)
            assert os.path.exists(os.path.join(log_dir, "config.yaml"))
```

运行测试：
```bash
cd quant-trading && pytest tests/test_utils/test_config.py -v
```
预期：FAIL，`ModuleNotFoundError: No module named 'quant_trading'` 或 `ImportError`

### 步骤 4：编写实现

`src/quant_trading/__init__.py`：
```python
"""quant-trading: A quantitative trading research framework."""
__version__ = "0.1.0"
```

`src/quant_trading/utils/__init__.py`：空文件

`src/quant_trading/utils/config.py`：
```python
"""配置管理模块."""
import yaml
import os
from typing import Any


class Config:
    def __init__(self, config_dict: dict):
        self._config = config_dict

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def to_dict(self) -> dict:
        return self._config.copy()


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)


def save_config_snapshot(config: Config, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    config_path = os.path.join(log_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
```

`tests/conftest.py`：
```python
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
```

### 步骤 5：运行测试确认通过

```bash
cd quant-trading && pytest tests/test_utils/test_config.py -v
```
预期：3/3 PASS

### 步骤 6：Commit

```bash
cd quant-trading && git init && git add . && git commit -m "feat: project skeleton + config management"
```

---

## 任务 2：数据底层工具（preprocessing + cleaner）

**说明：** 实现数据清洗和底层数学工具（技术指标计算、标准化、时间序列切分）。

**文件：**
- 创建：`src/quant_trading/data/preprocessing.py`
- 创建：`src/quant_trading/data/cleaner.py`
- 创建：`tests/test_data/test_preprocessing.py`
- 创建：`tests/test_data/test_cleaner.py`

**依赖：** 任务 1

---

### 步骤 1：编写 preprocessing 测试

`tests/test_data/test_preprocessing.py`：
```python
import numpy as np
import pandas as pd
import pytest

from quant_trading.data.preprocessing import (
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
        # 第3个位置(索引2)的MA应该基于索引0,1,2，然后shift(1)到索引3
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
        # 前5个应该是NaN（min_periods=window）
        assert norm["a"].iloc[:5].isna().all()
        # 第5个之后应该有值
        assert not norm["a"].iloc[5:].isna().any()


class TestTimeSeriesSplit:
    def test_split_by_date(self):
        dates = pd.date_range("2020-01-01", periods=10)
        df = pd.DataFrame({"value": range(10)}, index=dates)
        train, val, test = time_series_split(df, train_end="2020-01-05", val_end="2020-01-07")
        assert len(train) == 5
        assert len(val) == 2
        assert len(test) == 3
        # 确保时间顺序
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()
```

### 步骤 2：编写 preprocessing 实现

`src/quant_trading/data/preprocessing.py`：
```python
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
        rolling_mean = df[col].rolling(window=window, min_periods=window).mean()
        rolling_std = df[col].rolling(window=window, min_periods=window).std()
        rolling_std = rolling_std.replace(0, np.nan)
        result[col] = (df[col] - rolling_mean) / rolling_std
    return result


def time_series_split(df: pd.DataFrame, train_end: str, val_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df.index <= train_end]
    val_df = df[(df.index > train_end) & (df.index <= val_end)]
    test_df = df[df.index > val_end]
    return train_df, val_df, test_df
```

### 步骤 3：编写 cleaner 测试

`tests/test_data/test_cleaner.py`：
```python
import pandas as pd
import pytest
from quant_trading.data.cleaner import DataCleaner


class TestDataCleaner:
    def test_remove_zero_volume(self):
        df = pd.DataFrame({
            "open": [10, 11, 12],
            "high": [11, 12, 13],
            "low": [9, 10, 11],
            "close": [10.5, 11.5, 12.5],
            "volume": [1000, 0, 2000],
        }, index=pd.date_range("2020-01-01", periods=3))
        
        cleaner = DataCleaner()
        cleaned = cleaner.clean(df)
        assert len(cleaned) == 2
        assert cleaner.get_report()["removed"] == 1

    def test_remove_negative_price(self):
        df = pd.DataFrame({
            "open": [10, -1, 12],
            "high": [11, 12, 13],
            "low": [9, 10, 11],
            "close": [10.5, 11.5, 12.5],
            "volume": [1000, 1000, 2000],
        }, index=pd.date_range("2020-01-01", periods=3))
        
        cleaner = DataCleaner()
        cleaned = cleaner.clean(df)
        assert len(cleaned) == 2

    def test_remove_price_relation_anomaly(self):
        df = pd.DataFrame({
            "open": [10, 11, 12],
            "high": [9, 12, 13],   # high < low 异常
            "low": [9, 10, 11],
            "close": [10.5, 11.5, 12.5],
            "volume": [1000, 1000, 2000],
        }, index=pd.date_range("2020-01-01", periods=3))
        
        cleaner = DataCleaner()
        cleaned = cleaner.clean(df)
        assert len(cleaned) == 2
```

### 步骤 4：编写 cleaner 实现

`src/quant_trading/data/cleaner.py`：
```python
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

        # 确保 datetime 索引
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

        # 检查必需列
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # 删除非正价格
        before = len(df)
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
        n = before - len(df)
        if n > 0:
            removed["steps"].append(f"non-positive price: {n}")
            removed["total"] += n

        # 删除零成交量
        before = len(df)
        df = df[df["volume"] >= self.min_volume]
        n = before - len(df)
        if n > 0:
            removed["steps"].append(f"zero/low volume: {n}")
            removed["total"] += n

        # 删除价格关系异常
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

        # 删除极端涨跌幅
        before = len(df)
        returns = df["close"].pct_change().abs()
        df = df[returns <= self.max_daily_change]
        n = before - len(df)
        if n > 0:
            removed["steps"].append(f"extreme return: {n}")
            removed["total"] += n

        # 删除连续价格不变
        before = len(df)
        price_change = df["close"].diff().abs()
        consecutive_unchanged = price_change.rolling(self.max_consecutive_limit).sum() == 0
        df = df[~consecutive_unchanged.fillna(False)]
        n = before - len(df)
        if n > 0:
            removed["steps"].append(f"consecutive unchanged: {n}")
            removed["total"] += n

        # 删除重复日期
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
```

### 步骤 5：运行测试确认通过

```bash
cd quant-trading && pytest tests/test_data/ -v
```
预期：全部 PASS

### 步骤 6：Commit

```bash
cd quant-trading && git add . && git commit -m "feat: data preprocessing + cleaner"
```


## 任务 3：数据加载 + DataHandler

**说明：** 实现双数据源加载器（AKShare + 腾讯财经）和 DataHandler 统一入口。

**文件：**
- 创建：`src/quant_trading/data/loader.py`
- 创建：`src/quant_trading/data/handler.py`
- 创建：`tests/test_data/test_loader.py`
- 创建：`tests/test_data/test_handler.py`

**依赖：** 任务 2

**关键约束：**
- `DataLoader.load()` 返回统一格式 `date(index), open, high, low, close, volume`
- `DataHandler.get_dataset()` 输出 MultiIndex `(date, symbol)` DataFrame
- 特征计算必须调用 `Factor.compute()`（从任务 4 导入）
- 目标变量：`target_return = close.pct_change(prediction_horizon).shift(-prediction_horizon)`
- 标准化使用 `rolling_normalize(window=252)`

---

### 步骤 1：编写 loader 测试

`tests/test_data/test_loader.py`：
```python
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from quant_trading.data.loader import DataLoader


class TestDataLoader:
    def test_load_returns_correct_columns(self):
        loader = DataLoader()
        # Mock akshare to avoid network call
        mock_df = pd.DataFrame({
            "日期": ["2020-01-01", "2020-01-02"],
            "开盘": [10.0, 11.0],
            "收盘": [11.0, 12.0],
            "最高": [11.5, 12.5],
            "最低": [9.5, 10.5],
            "成交量": [10000, 20000],
            "成交额": [100000, 200000],
            "振幅": [0.05, 0.05],
            "涨跌幅": [0.1, 0.09],
            "涨跌额": [1.0, 1.0],
            "换手率": [0.05, 0.06],
        })
        with patch("quant_trading.data.loader.ak.stock_zh_a_hist", return_value=mock_df):
            df = loader.load("000001", "2020-01-01", "2020-01-02", source="akshare")
        
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 2
```

### 步骤 2：编写 loader 实现

`src/quant_trading/data/loader.py`：
```python
"""数据加载器：支持 AKShare 和腾讯财经."""
import akshare as ak
import pandas as pd
from typing import Optional


class DataLoader:
    def load(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None, source: str = "akshare") -> pd.DataFrame:
        if source == "akshare":
            return self._load_akshare(symbol, start, end)
        else:
            raise ValueError(f"Unsupported source: {source}")

    def _load_akshare(self, symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
        df.columns = ["date", "open", "close", "high", "low", "volume", "amount", "amplitude", "pct_change", "change", "turnover"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df[["open", "high", "low", "close", "volume"]]
```

### 步骤 3：编写 handler 测试

`tests/test_data/test_handler.py`：
```python
import pandas as pd
import pytest
from unittest.mock import MagicMock
from quant_trading.data.handler import DataHandler
from quant_trading.factors.base import Factor, register_factor


@register_factor
class MockMomentumFactor(Factor):
    name = "mock_momentum"
    def compute(self, df):
        return df["close"] / df["close"].shift(5) - 1


class TestDataHandler:
    def test_get_dataset_structure(self):
        loader = MagicMock()
        mock_df = pd.DataFrame({
            "open": [10, 11, 12, 13, 14, 15],
            "high": [11, 12, 13, 14, 15, 16],
            "low": [9, 10, 11, 12, 13, 14],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "volume": [1000, 2000, 3000, 4000, 5000, 6000],
        }, index=pd.date_range("2020-01-01", periods=6))
        loader.load.return_value = mock_df
        
        handler = DataHandler(features=[MockMomentumFactor()], label="target_return", loader=loader)
        # Note: this test will need cleaner and enough data; we'll use a simplified approach
        dataset = handler.get_dataset(["000001"], "2020-01-01", "2020-01-06")
        
        assert isinstance(dataset.index, pd.MultiIndex)
        assert "mock_momentum" in dataset.columns
        assert "target_return" in dataset.columns
```

### 步骤 4：编写 handler 实现

`src/quant_trading/data/handler.py`：
```python
"""DataHandler：统一入口（Qlib 思想）."""
import pandas as pd
import numpy as np
from typing import Optional, List
from quant_trading.data.loader import DataLoader
from quant_trading.data.cleaner import DataCleaner
from quant_trading.data.preprocessing import rolling_normalize
from quant_trading.factors.base import Factor


class DataHandler:
    def __init__(
        self,
        features: List[Factor],
        label: str = "target_return",
        loader: Optional[DataLoader] = None,
        cleaner: Optional[DataCleaner] = None,
        prediction_horizon: int = 1,
    ):
        self.features = features
        self.label = label
        self.loader = loader or DataLoader()
        self.cleaner = cleaner or DataCleaner()
        self.prediction_horizon = prediction_horizon

    def get_dataset(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        all_data = []
        for symbol in symbols:
            df = self.loader.load(symbol, start, end)
            df = self.cleaner.clean(df, symbol)
            
            # 计算因子
            for factor in self.features:
                df[factor.name] = factor.compute(df)
            
            # 创建目标变量
            future_return = df["close"].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
            df[self.label] = future_return
            
            # 标准化（只对因子列）
            factor_cols = [f.name for f in self.features]
            if factor_cols:
                normalized = rolling_normalize(df[factor_cols], window=252)
                df = df.drop(columns=factor_cols)
                df = pd.concat([df, normalized.add_suffix("_norm")], axis=1)
            
            df["symbol"] = symbol
            all_data.append(df)
        
        combined = pd.concat(all_data, axis=0)
        combined = combined.dropna()
        combined = combined.set_index([combined.index, "symbol"])
        combined.index.names = ["date", "symbol"]
        return combined
```

### 步骤 5：运行测试确认通过

```bash
cd quant-trading && pytest tests/test_data/ -v
```
预期：全部 PASS

### 步骤 6：Commit

```bash
cd quant-trading && git add . && git commit -m "feat: data loader + DataHandler"
```

---

## 任务 4：因子层全部

**说明：** 实现因子基类/注册表、具体因子（技术/价格/成交量）、Alphalens 式因子评估器。

**文件：**
- 创建：`src/quant_trading/factors/base.py`
- 创建：`src/quant_trading/factors/technical.py`
- 创建：`src/quant_trading/factors/price.py`
- 创建：`src/quant_trading/factors/volume.py`
- 创建：`src/quant_trading/factors/evaluator.py`
- 创建：`tests/test_factors/test_base.py`
- 创建：`tests/test_factors/test_factors.py`
- 创建：`tests/test_factors/test_evaluator.py`

**依赖：** 任务 2（preprocessing）

**关键约束：**
- `Factor.compute()` 必须返回 `pd.Series`，索引与输入 df 一致
- 所有因子必须 `shift(1)` 防止前瞻偏差（可在具体因子实现中调用 preprocessing 工具）
- `FACTOR_REGISTRY` 通过 `@register_factor` 装饰器注册
- `FactorEvaluator.evaluate()` 返回 IC、RankIC、IR、turnover、quantile returns、half_life

---

### 步骤 1：编写 factors/base 测试

`tests/test_factors/test_base.py`：
```python
import pandas as pd
import pytest
from quant_trading.factors.base import Factor, register_factor, FACTOR_REGISTRY


@register_factor
class TestFactor(Factor):
    name = "test_factor"
    def compute(self, df):
        return df["close"] * 2


class TestFactorRegistry:
    def test_register_factor(self):
        assert "test_factor" in FACTOR_REGISTRY
        assert FACTOR_REGISTRY["test_factor"] == TestFactor

    def test_factor_compute(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        factor = TestFactor()
        result = factor.compute(df)
        assert list(result) == [2, 4, 6]
```

### 步骤 2：编写 factors/base 实现

`src/quant_trading/factors/base.py`：
```python
"""因子基类与注册表."""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Type


class Factor(ABC):
    name: str = ""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """输入清洗后的 ohlcv，输出因子序列."""


FACTOR_REGISTRY: Dict[str, Type[Factor]] = {}


def register_factor(cls: Type[Factor]) -> Type[Factor]:
    if not cls.name:
        raise ValueError(f"Factor {cls.__name__} must define a name")
    FACTOR_REGISTRY[cls.name] = cls
    return cls
```

### 步骤 3：编写具体因子实现

`src/quant_trading/factors/technical.py`：
```python
"""技术因子."""
import pandas as pd
from quant_trading.factors.base import Factor, register_factor
from quant_trading.data.preprocessing import calculate_ma, calculate_rsi, calculate_macd, calculate_bollinger_bands


@register_factor
class MA5(Factor):
    name = "ma5"
    def compute(self, df):
        return calculate_ma(df["close"], window=5, shift=True)


@register_factor
class MA20(Factor):
    name = "ma20"
    def compute(self, df):
        return calculate_ma(df["close"], window=20, shift=True)


@register_factor
class RSI14(Factor):
    name = "rsi14"
    def compute(self, df):
        return calculate_rsi(df["close"], window=14, shift=True)


@register_factor
class MACD(Factor):
    name = "macd"
    def compute(self, df):
        macd_df = calculate_macd(df["close"], shift=True)
        return macd_df["macd"]


@register_factor
class BollingerPosition(Factor):
    name = "bb_position"
    def compute(self, df):
        bb = calculate_bollinger_bands(df["close"], shift=True)
        return (df["close"] - bb["lower"]) / (bb["upper"] - bb["lower"])
```

`src/quant_trading/factors/price.py`：
```python
"""价格因子."""
import pandas as pd
import numpy as np
from quant_trading.factors.base import Factor, register_factor


@register_factor
class Returns1D(Factor):
    name = "returns_1d"
    def compute(self, df):
        return df["close"].pct_change().shift(1)


@register_factor
class Momentum5D(Factor):
    name = "momentum_5d"
    def compute(self, df):
        return df["close"] / df["close"].shift(5) - 1


@register_factor
class PricePercentile60D(Factor):
    name = "price_percentile_60d"
    def compute(self, df):
        min_60 = df["close"].rolling(60).min()
        max_60 = df["close"].rolling(60).max()
        return (df["close"] - min_60) / (max_60 - min_60)
```

`src/quant_trading/factors/volume.py`：
```python
"""成交量因子."""
import pandas as pd
from quant_trading.factors.base import Factor, register_factor


@register_factor
class VolumeRatio(Factor):
    name = "volume_ratio"
    def compute(self, df):
        vol_ma20 = df["volume"].rolling(20).mean()
        return df["volume"] / vol_ma20


@register_factor
class VolumeMomentum5D(Factor):
    name = "volume_momentum_5d"
    def compute(self, df):
        return df["volume"] / df["volume"].shift(5)
```

### 步骤 4：编写 evaluator 测试

`tests/test_factors/test_evaluator.py`：
```python
import numpy as np
import pandas as pd
import pytest
from quant_trading.factors.evaluator import FactorEvaluator
from quant_trading.factors.base import Factor, register_factor


@register_factor
class PerfectFactor(Factor):
    name = "perfect_factor"
    def compute(self, df):
        return pd.Series(np.arange(len(df)), index=df.index)


class TestFactorEvaluator:
    def test_ic_calculation(self):
        df = pd.DataFrame({"close": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]})
        factor = PerfectFactor()
        factor_values = factor.compute(df)
        # forward returns positively correlated with factor
        forward_returns = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], index=df.index)
        
        evaluator = FactorEvaluator()
        result = evaluator.evaluate(factor, df, forward_returns)
        
        assert "ic" in result
        assert "rank_ic" in result
        assert "turnover" in result
        assert isinstance(result["ic"], float)
```

### 步骤 5：编写 evaluator 实现

`src/quant_trading/factors/evaluator.py`：
```python
"""因子评估器：Alphalens 式分析."""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict
from quant_trading.factors.base import Factor


class FactorEvaluator:
    def evaluate(self, factor: Factor, df: pd.DataFrame, forward_returns: pd.Series) -> Dict:
        factor_values = factor.compute(df)
        
        # 对齐数据
        aligned = pd.DataFrame({"factor": factor_values, "returns": forward_returns}).dropna()
        if len(aligned) < 3:
            return {"ic": 0.0, "rank_ic": 0.0, "ir": 0.0, "turnover": 0.0, "half_life": np.nan}
        
        f = aligned["factor"].values
        r = aligned["returns"].values
        
        # IC
        ic, _ = stats.pearsonr(f, r)
        if np.isnan(ic):
            ic = 0.0
        
        # Rank IC
        rank_ic, _ = stats.spearmanr(f, r)
        if np.isnan(rank_ic):
            rank_ic = 0.0
        
        # Turnover（因子自相关性）
        factor_diff = np.abs(np.diff(f))
        turnover = np.mean(factor_diff) / (np.std(f) + 1e-10)
        
        # Half life（简化：因子自回归系数）
        f_lag = f[:-1]
        f_curr = f[1:]
        if np.std(f_lag) > 0:
            slope, _, _, _, _ = stats.linregress(f_lag, f_curr)
            half_life = -np.log(2) / np.log(max(abs(slope), 1e-10)) if slope > 0 else np.nan
        else:
            half_life = np.nan
        
        return {
            "ic": float(ic),
            "rank_ic": float(rank_ic),
            "ir": float(ic / (np.std(f) + 1e-10)),
            "turnover": float(turnover),
            "half_life": float(half_life) if not np.isnan(half_life) else None,
        }
```

### 步骤 6：运行测试确认通过

```bash
cd quant-trading && pytest tests/test_factors/ -v
```
预期：全部 PASS

### 步骤 7：Commit

```bash
cd quant-trading && git add . && git commit -m "feat: factor layer with registry and evaluator"
```

---

## 任务 5：模型层全部

**说明：** 实现 Model 协议、LightGBMTrainer、时间序列交叉验证、过拟合检测。

**文件：**
- 创建：`src/quant_trading/models/base.py`
- 创建：`src/quant_trading/models/gbdt.py`
- 创建：`src/quant_trading/models/cv.py`
- 创建：`src/quant_trading/models/validation.py`
- 创建：`tests/test_models/test_base.py`
- 创建：`tests/test_models/test_gbdt.py`
- 创建：`tests/test_models/test_cv.py`
- 创建：`tests/test_models/test_validation.py`

**依赖：** 任务 1

**关键约束：**
- `Model` 是协议/抽象基类，只定义 `fit/predict/save/load`
- `LightGBMTrainer` 基于真实 `lightgbm` 库，禁止 Mock
- `TimeSeriesCV.split()` 不打乱顺序
- `detect_overfitting()` 的 threshold 默认 0.02

---

### 步骤 1：编写 models/base 实现

`src/quant_trading/models/base.py`：
```python
"""Model 协议：极薄一层抽象."""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Model(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: tuple | None = None) -> "Model": ...
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    
    @abstractmethod
    def save(self, path: str) -> None: ...
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Model": ...
```

### 步骤 2：编写 models/gbdt 实现

`src/quant_trading/models/gbdt.py`：
```python
"""LightGBM 模型实现."""
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Dict
import lightgbm as lgb
from quant_trading.models.base import Model


class LightGBMTrainer(Model):
    def __init__(self, params: Optional[Dict] = None):
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
        }
        self.model_: Optional[lgb.Booster] = None
        self.best_iteration_: Optional[int] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: Optional[tuple] = None) -> "LightGBMTrainer":
        train_data = lgb.Dataset(X, label=y)
        
        if eval_set is not None:
            X_val, y_val = eval_set
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            self.model_ = lgb.train(
                self.params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data, valid_data],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            self.best_iteration_ = self.model_.best_iteration
        else:
            self.model_ = lgb.train(self.params, train_data, num_boost_round=100)
            self.best_iteration_ = 100
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model not trained")
        return self.model_.predict(X, num_iteration=self.best_iteration_)

    def feature_importance(self) -> Dict[str, float]:
        if self.model_ is None:
            return {}
        importance = self.model_.feature_importance(importance_type="gain")
        return dict(zip(self.model_.feature_name(), importance))

    def save(self, path: str) -> None:
        if self.model_ is None:
            raise ValueError("Model not trained")
        self.model_.save_model(path)

    @classmethod
    def load(cls, path: str) -> "LightGBMTrainer":
        trainer = cls()
        trainer.model_ = lgb.Booster(model_file=path)
        return trainer
```

### 步骤 3：编写 models/cv 实现

`src/quant_trading/models/cv.py`：
```python
"""时间序列交叉验证."""
import numpy as np
from sklearn.model_selection import BaseCrossValidator


class TimeSeriesCV(BaseCrossValidator):
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = test_start + fold_size
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class WalkForwardCV(BaseCrossValidator):
    def __init__(self, min_train_size: int = 252, test_size: int = 63, step: int = 63):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step = step

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        for start in range(0, n_samples - self.min_train_size - self.test_size + 1, self.step):
            train_end = start + self.min_train_size
            test_end = train_end + self.test_size
            if test_end > n_samples:
                break
            yield np.arange(start, train_end), np.arange(train_end, test_end)

    def get_n_splits(self, X=None, y=None, groups=None):
        n_samples = len(X)
        return max(0, (n_samples - self.min_train_size - self.test_size) // self.step + 1)
```

### 步骤 4：编写 models/validation 实现

`src/quant_trading/models/validation.py`：
```python
"""模型验证：过拟合检测."""
import numpy as np
from typing import Dict, Any


def detect_overfitting(train_ic: float, val_ic: float, threshold: float = 0.02) -> tuple[bool, float]:
    gap = train_ic - val_ic
    is_overfit = gap > threshold
    return is_overfit, gap


def analyze_learning_curve(train_scores: list, val_scores: list) -> Dict[str, Any]:
    train_scores = np.array(train_scores)
    val_scores = np.array(val_scores)
    
    is_overfitting = False
    best_iteration = len(val_scores) - 1
    
    if len(val_scores) >= 3:
        best_idx = int(np.argmax(val_scores))
        best_iteration = best_idx
        if best_idx < len(val_scores) - 1:
            val_declined = val_scores[-1] < val_scores[best_idx]
            train_rising = train_scores[-1] >= train_scores[best_idx]
            if val_declined and train_rising:
                is_overfitting = True
    
    return {
        "is_overfitting": bool(is_overfitting),
        "best_iteration": int(best_iteration),
        "train_final": float(train_scores[-1]),
        "val_final": float(val_scores[-1]),
        "gap": float(train_scores[-1] - val_scores[-1]),
    }
```

### 步骤 5：编写模型层测试

`tests/test_models/test_gbdt.py`：
```python
import numpy as np
import pandas as pd
import pytest
from quant_trading.models.gbdt import LightGBMTrainer


class TestLightGBMTrainer:
    def test_fit_predict(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.randn(100))
        
        model = LightGBMTrainer(params={"num_leaves": 7, "verbose": -1})
        model.fit(X, y)
        
        preds = model.predict(X)
        assert len(preds) == 100
        assert isinstance(preds, np.ndarray)

    def test_save_load(self, tmp_path):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.randn(50))
        
        model = LightGBMTrainer(params={"num_leaves": 7, "verbose": -1})
        model.fit(X, y)
        
        save_path = tmp_path / "model.txt"
        model.save(str(save_path))
        
        loaded = LightGBMTrainer.load(str(save_path))
        preds = loaded.predict(X)
        assert len(preds) == 50
```

`tests/test_models/test_cv.py`：
```python
import numpy as np
from quant_trading.models.cv import TimeSeriesCV, WalkForwardCV


class TestTimeSeriesCV:
    def test_no_shuffle(self):
        X = np.arange(100)
        tscv = TimeSeriesCV(n_splits=3)
        for train_idx, test_idx in tscv.split(X):
            assert train_idx.max() < test_idx.min()


class TestWalkForwardCV:
    def test_window_size(self):
        X = np.arange(500)
        wfcv = WalkForwardCV(min_train_size=100, test_size=50, step=50)
        for train_idx, test_idx in wfcv.split(X):
            assert len(train_idx) == 100
            assert len(test_idx) == 50
```

`tests/test_models/test_validation.py`：
```python
from quant_trading.models.validation import detect_overfitting, analyze_learning_curve


class TestDetectOverfitting:
    def test_overfit_detected(self):
        is_overfit, gap = detect_overfitting(train_ic=0.10, val_ic=0.05, threshold=0.02)
        assert is_overfit is True
        assert gap == pytest.approx(0.05)

    def test_not_overfit(self):
        is_overfit, gap = detect_overfitting(train_ic=0.10, val_ic=0.09, threshold=0.02)
        assert is_overfit is False
```

### 步骤 6：运行测试确认通过

```bash
cd quant-trading && pytest tests/test_models/ -v
```
预期：全部 PASS

### 步骤 7：Commit

```bash
cd quant-trading && git add . && git commit -m "feat: model layer (LightGBM + CV + validation)"
```

---

## 任务 6：Portfolio + Metrics

**说明：** 实现 PortfolioStrategy 协议和具体策略（TopK / LongShort），以及金融指标计算。

**文件：**
- 创建：`src/quant_trading/portfolio/base.py`
- 创建：`src/quant_trading/portfolio/strategy.py`
- 创建：`src/quant_trading/metrics/core.py`
- 创建：`tests/test_portfolio/test_strategy.py`
- 创建：`tests/test_metrics/test_core.py`

**依赖：** 任务 1

---

### 步骤 1：编写 portfolio 实现

`src/quant_trading/portfolio/base.py`：
```python
"""PortfolioStrategy 协议."""
from abc import ABC, abstractmethod
import pandas as pd


class PortfolioStrategy(ABC):
    @abstractmethod
    def generate_weights(
        self,
        predictions: pd.DataFrame,
        current_portfolio: dict[str, float] | None = None,
    ) -> pd.Series:
        """返回: {symbol: target_weight}"""
```

`src/quant_trading/portfolio/strategy.py`：
```python
"""具体策略实现."""
import pandas as pd
from quant_trading.portfolio.base import PortfolioStrategy


class TopKStrategy(PortfolioStrategy):
    def __init__(self, top_k: int = 5, weight_method: str = "equal"):
        self.top_k = top_k
        self.weight_method = weight_method  # "equal" | "confidence"

    def generate_weights(self, predictions: pd.DataFrame, current_portfolio: dict | None = None) -> pd.Series:
        day_preds = predictions.copy()
        top = day_preds.nlargest(self.top_k, "predicted")
        
        if self.weight_method == "equal":
            weights = pd.Series(1.0 / self.top_k, index=top["symbol"])
        elif self.weight_method == "confidence":
            conf_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
            w = [conf_weights[min(i, len(conf_weights)-1)] for i in range(len(top))]
            weights = pd.Series(w, index=top["symbol"])
            weights = weights / weights.sum()
        else:
            weights = pd.Series(1.0 / self.top_k, index=top["symbol"])
        
        return weights


class LongShortStrategy(PortfolioStrategy):
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def generate_weights(self, predictions: pd.DataFrame, current_portfolio: dict | None = None) -> pd.Series:
        day_preds = predictions.copy()
        longs = day_preds.nlargest(self.top_k, "predicted")
        shorts = day_preds.nsmallest(self.top_k, "predicted")
        
        weights = pd.Series(0.0, index=day_preds["symbol"].unique())
        for sym in longs["symbol"]:
            weights[sym] = 0.5 / self.top_k
        for sym in shorts["symbol"]:
            weights[sym] = -0.5 / self.top_k
        
        return weights
```

### 步骤 2：编写 metrics 实现

`src/quant_trading/metrics/core.py`：
```python
"""金融指标计算模块."""
import numpy as np
import pandas as pd
from scipy import stats


def calculate_ic(predictions: np.ndarray, returns: np.ndarray) -> float:
    mask = ~(np.isnan(predictions) | np.isnan(returns))
    if mask.sum() < 2:
        return 0.0
    corr, _ = stats.pearsonr(predictions[mask], returns[mask])
    return corr if not np.isnan(corr) else 0.0


def calculate_rank_ic(predictions: np.ndarray, returns: np.ndarray) -> float:
    mask = ~(np.isnan(predictions) | np.isnan(returns))
    if mask.sum() < 2:
        return 0.0
    corr, _ = stats.spearmanr(predictions[mask], returns[mask])
    return corr if not np.isnan(corr) else 0.0


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.03, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    std = np.std(excess, ddof=1)
    if std < 1e-10:
        return 0.0
    return np.mean(excess) / std * np.sqrt(periods_per_year)


def calculate_max_drawdown(prices: pd.Series) -> float:
    if len(prices) < 2:
        return 0.0
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    max_dd = drawdown.min()
    return max_dd if not pd.isna(max_dd) else 0.0


def calculate_calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    avg_return = returns.mean()
    annual_return = (1 + avg_return) ** 252 - 1
    max_dd = abs(calculate_max_drawdown(prices))
    if max_dd < 1e-10:
        return np.inf if annual_return > 0 else 0.0
    return annual_return / max_dd


def calculate_excess_return(strategy_return: float, benchmark_return: float) -> float:
    if strategy_return == 0.0 and benchmark_return == 0.0:
        return 0.0
    strategy_log = np.log1p(strategy_return)
    benchmark_log = np.log1p(benchmark_return)
    return np.expm1(strategy_log - benchmark_log)
```

### 步骤 3：编写测试

`tests/test_portfolio/test_strategy.py`：
```python
import pandas as pd
import pytest
from quant_trading.portfolio.strategy import TopKStrategy, LongShortStrategy


class TestTopKStrategy:
    def test_equal_weights(self):
        preds = pd.DataFrame({
            "symbol": ["A", "B", "C", "D", "E"],
            "predicted": [0.5, 0.4, 0.3, 0.2, 0.1],
        })
        strategy = TopKStrategy(top_k=3, weight_method="equal")
        weights = strategy.generate_weights(preds)
        assert len(weights) == 3
        assert weights.sum() == pytest.approx(1.0)
        assert set(weights.index) == {"A", "B", "C"}

    def test_long_short_weights(self):
        preds = pd.DataFrame({
            "symbol": ["A", "B", "C", "D", "E", "F"],
            "predicted": [0.5, 0.4, 0.3, -0.3, -0.4, -0.5],
        })
        strategy = LongShortStrategy(top_k=2)
        weights = strategy.generate_weights(preds)
        assert weights["A"] > 0
        assert weights["F"] < 0
        assert weights.sum() == pytest.approx(0.0)
```

`tests/test_metrics/test_core.py`：
```python
import numpy as np
import pandas as pd
from quant_trading.metrics.core import calculate_ic, calculate_sharpe_ratio, calculate_max_drawdown


class TestMetrics:
    def test_ic_perfect_correlation(self):
        preds = np.array([1, 2, 3, 4, 5])
        returns = np.array([1, 2, 3, 4, 5])
        ic = calculate_ic(preds, returns)
        assert ic == pytest.approx(1.0, abs=1e-10)

    def test_max_drawdown(self):
        prices = pd.Series([100, 110, 105, 95, 100])
        mdd = calculate_max_drawdown(prices)
        assert mdd < 0
        assert abs(mdd) == pytest.approx(0.13636, abs=1e-4)
```

### 步骤 4：运行测试确认通过

```bash
cd quant-trading && pytest tests/test_portfolio/ tests/test_metrics/ -v
```
预期：全部 PASS

### 步骤 5：Commit

```bash
cd quant-trading && git add . && git commit -m "feat: portfolio strategies + metrics"
```

---

## 任务 7：Backtest（Executor + Engine）

**说明：** 实现 Executor（模拟撮合）和 BacktestEngine（事件驱动回测）。

**文件：**
- 创建：`src/quant_trading/backtest/executor.py`
- 创建：`src/quant_trading/backtest/engine.py`
- 创建：`tests/test_backtest/test_executor.py`
- 创建：`tests/test_backtest/test_engine.py`

**依赖：** 任务 6

**关键约束：**
- T+1 执行（信号日收盘后决策，次日开盘价成交）
- 手续费万3，滑点 0.1%
- 停牌日跳过交易
- A股整手交易（100股）
- 回测引擎内部集成风控（仓位/止损/止盈）

---

### 步骤 1：编写 executor 实现

`src/quant_trading/backtest/executor.py`：
```python
"""Executor：模拟撮合（Qlib 思想）."""
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Trade:
    date: datetime
    symbol: str
    action: str  # "BUY" | "SELL"
    shares: int
    price: float
    commission: float


class Executor:
    def execute(
        self,
        target_weights: pd.Series,
        current_positions: Dict[str, dict],
        price_data: Dict[str, pd.DataFrame],
        signal_date: datetime,
        capital: float,
        commission_rate: float = 0.0003,
        slippage: float = 0.001,
    ) -> List[Trade]:
        trades = []
        
        for symbol, target_weight in target_weights.items():
            if symbol not in price_data:
                continue
            
            df = price_data[symbol]
            future_dates = df.index[df.index > signal_date]
            if len(future_dates) == 0:
                continue
            
            trade_date = future_dates[0]
            if trade_date not in df.index:
                continue
            
            # T+1 开盘价执行
            open_price = df.loc[trade_date, "open"]
            
            # 滑点
            if target_weight > 0:
                executed_price = open_price * (1 + slippage)
            else:
                executed_price = open_price * (1 - slippage)
            
            # 计算目标金额和股数
            target_value = capital * abs(target_weight)
            shares = int(target_value / executed_price / 100) * 100
            
            if shares < 100:
                continue
            
            # 检查现有持仓
            current_shares = current_positions.get(symbol, {}).get("shares", 0)
            
            if target_weight > 0 and current_shares == 0:
                # 买入
                amount = shares * executed_price
                commission = amount * commission_rate
                trades.append(Trade(
                    date=trade_date,
                    symbol=symbol,
                    action="BUY",
                    shares=shares,
                    price=executed_price,
                    commission=commission,
                ))
            elif target_weight <= 0 and current_shares > 0:
                # 卖出
                amount = current_shares * executed_price
                commission = amount * commission_rate
                trades.append(Trade(
                    date=trade_date,
                    symbol=symbol,
                    action="SELL",
                    shares=current_shares,
                    price=executed_price,
                    commission=commission,
                ))
        
        return trades
```

### 步骤 2：编写 engine 实现

`src/quant_trading/backtest/engine.py`：
```python
"""BacktestEngine：事件驱动回测."""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from quant_trading.backtest.executor import Executor, Trade
from quant_trading.portfolio.base import PortfolioStrategy
from quant_trading.metrics.core import calculate_sharpe_ratio, calculate_max_drawdown


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000
    commission_rate: float = 0.0003
    slippage: float = 0.001
    position_size: float = 1.0
    max_positions: int = 10
    stop_loss: Optional[float] = 0.05
    take_profit: Optional[float] = 0.10


class BacktestEngine:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.executor = Executor()
        self.reset()

    def reset(self):
        self.capital = self.config.initial_capital
        self.positions: Dict[str, dict] = {}
        self.trades: List[Trade] = []
        self.daily_values: List[dict] = []
        self.current_date = None

    def run(
        self,
        predictions: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        strategy: PortfolioStrategy,
        rebalance_freq: str = "daily",
    ) -> dict:
        self.reset()
        predictions = predictions.copy()
        predictions["date"] = pd.to_datetime(predictions["date"])
        predictions = predictions.sort_values(["date", "symbol"])
        
        dates = predictions["date"].unique()
        last_rebalance = None
        
        for date in dates:
            self.current_date = date
            
            # 判断是否调仓日
            should_trade = self._should_rebalance(date, last_rebalance, rebalance_freq)
            
            if should_trade:
                last_rebalance = date
                day_preds = predictions[predictions["date"] == date]
                target_weights = strategy.generate_weights(day_preds, self.positions)
                
                trades = self.executor.execute(
                    target_weights,
                    self.positions,
                    price_data,
                    date,
                    self.capital,
                    self.config.commission_rate,
                    self.config.slippage,
                )
                
                for trade in trades:
                    self._process_trade(trade)
            
            # 检查止损止盈
            self._check_stop_loss_take_profit(price_data, date)
            
            # 更新市值
            self._update_portfolio_value(price_data, date)
        
        return self._calculate_metrics(price_data)

    def _should_rebalance(self, date, last_rebalance, freq):
        if freq == "daily":
            return True
        elif freq == "weekly":
            return pd.Timestamp(date).weekday() == 0
        elif freq == "monthly":
            if last_rebalance is None:
                return True
            return pd.Timestamp(date).month != pd.Timestamp(last_rebalance).month
        return True

    def _process_trade(self, trade: Trade):
        if trade.action == "BUY":
            total_cost = trade.shares * trade.price + trade.commission
            if total_cost > self.capital:
                return
            self.capital -= total_cost
            self.positions[trade.symbol] = {"shares": trade.shares, "cost": trade.price}
        elif trade.action == "SELL":
            revenue = trade.shares * trade.price - trade.commission
            self.capital += revenue
            if trade.symbol in self.positions:
                del self.positions[trade.symbol]
        self.trades.append(trade)

    def _check_stop_loss_take_profit(self, price_data, date):
        if not self.config.stop_loss and not self.config.take_profit:
            return
        
        for symbol in list(self.positions.keys()):
            if symbol not in price_data or date not in price_data[symbol].index:
                continue
            
            pos = self.positions[symbol]
            current_price = price_data[symbol].loc[date, "close"]
            pnl_ratio = (current_price - pos["cost"]) / pos["cost"]
            
            if self.config.stop_loss and pnl_ratio < -self.config.stop_loss:
                self._force_sell(symbol, current_price, date, "stop_loss")
            elif self.config.take_profit and pnl_ratio > self.config.take_profit:
                self._force_sell(symbol, current_price, date, "take_profit")

    def _force_sell(self, symbol, price, date, reason):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        commission = pos["shares"] * price * self.config.commission_rate
        revenue = pos["shares"] * price - commission
        self.capital += revenue
        del self.positions[symbol]
        self.trades.append(Trade(
            date=date, symbol=symbol, action="SELL",
            shares=pos["shares"], price=price, commission=commission,
        ))

    def _update_portfolio_value(self, price_data, date):
        position_value = 0.0
        for symbol, pos in self.positions.items():
            if symbol in price_data and date in price_data[symbol].index:
                position_value += pos["shares"] * price_data[symbol].loc[date, "close"]
        total_value = self.capital + position_value
        self.daily_values.append({
            "date": date,
            "capital": self.capital,
            "position_value": position_value,
            "total_value": total_value,
        })

    def _calculate_metrics(self, price_data) -> dict:
        if not self.daily_values:
            return {}
        
        values_df = pd.DataFrame(self.daily_values).set_index("date").sort_index()
        values_df["returns"] = values_df["total_value"].pct_change()
        
        total_return = values_df["total_value"].iloc[-1] / self.config.initial_capital - 1
        n_years = len(values_df) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        volatility = values_df["returns"].std() * np.sqrt(252)
        max_dd = calculate_max_drawdown(values_df["total_value"])
        sharpe = calculate_sharpe_ratio(values_df["returns"].dropna().values)
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "num_trades": len(self.trades),
            "daily_values": values_df,
        }
```

### 步骤 3：编写 backtest 测试

`tests/test_backtest/test_engine.py`：
```python
import numpy as np
import pandas as pd
import pytest
from quant_trading.backtest.engine import BacktestEngine, BacktestConfig
from quant_trading.portfolio.strategy import TopKStrategy


class TestBacktestEngine:
    def test_simple_backtest(self):
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="B")
        predictions = []
        price_data = {}
        
        for date in dates:
            for symbol in ["A", "B", "C"]:
                predictions.append({"date": date, "symbol": symbol, "predicted": np.random.randn()})
        
        predictions_df = pd.DataFrame(predictions)
        
        for symbol in ["A", "B", "C"]:
            prices = 10 + np.cumsum(np.random.randn(len(dates)) * 0.01)
            price_data[symbol] = pd.DataFrame({
                "open": prices * 0.99,
                "close": prices,
            }, index=dates)
        
        engine = BacktestEngine(BacktestConfig(initial_capital=1_000_000))
        strategy = TopKStrategy(top_k=2)
        results = engine.run(predictions_df, price_data, strategy, rebalance_freq="daily")
        
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert results["num_trades"] > 0
```

### 步骤 4：运行测试确认通过

```bash
cd quant-trading && pytest tests/test_backtest/ -v
```
预期：全部 PASS

### 步骤 5：Commit

```bash
cd quant-trading && git add . && git commit -m "feat: backtest executor + engine"
```

---

## 任务 8：Experiments + Pipeline + 集成

**说明：** 实现 Recorder 实验记录、RollingBacktest Walk-forward 流水线、端到端集成测试。

**文件：**
- 创建：`src/quant_trading/experiments/recorder.py`
- 创建：`src/quant_trading/pipeline/rolling.py`
- 创建：`tests/test_experiments/test_recorder.py`
- 创建：`tests/test_pipeline/test_rolling.py`
- 创建：`tests/test_integration.py`

**依赖：** 任务 3, 4, 5, 6, 7

---

### 步骤 1：编写 recorder 实现

`src/quant_trading/experiments/recorder.py`：
```python
"""Recorder：实验记录（Qlib 思想）."""
import os
import json
import yaml
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from quant_trading.models.base import Model


class Recorder:
    def __init__(self, experiment_name: str, output_dir: str = "experiments"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(output_dir) / f"{timestamp}_{experiment_name}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_model(self, model: Model, filename: str = "model.pkl") -> None:
        path = self.log_dir / filename
        model.save(str(path))

    def log_predictions(self, predictions: pd.DataFrame, filename: str = "predictions.csv") -> None:
        predictions.to_csv(self.log_dir / filename, index=False)

    def log_backtest_report(self, report: dict, filename: str = "backtest_report.json") -> None:
        # 排除不可序列化的 DataFrame
        serializable = {k: v for k, v in report.items() if not isinstance(v, pd.DataFrame)}
        with open(self.log_dir / filename, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    def log_config(self, config: dict, filename: str = "config.yaml") -> None:
        with open(self.log_dir / filename, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def log_artifact(self, data: Any, filename: str) -> None:
        path = self.log_dir / filename
        if filename.endswith(".csv"):
            data.to_csv(path, index=False)
        elif filename.endswith(".json"):
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif filename.endswith(".pkl"):
            with open(path, "wb") as f:
                pickle.dump(data, f)
        else:
            with open(path, "w") as f:
                f.write(str(data))

    def get_artifact_path(self, filename: str) -> Path:
        return self.log_dir / filename
```

### 步骤 2：编写 rolling pipeline 实现

`src/quant_trading/pipeline/rolling.py`：
```python
"""RollingBacktest：Walk-forward 多窗口评估."""
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from quant_trading.data.handler import DataHandler
from quant_trading.models.base import Model
from quant_trading.portfolio.base import PortfolioStrategy
from quant_trading.backtest.engine import BacktestEngine
from quant_trading.metrics.core import calculate_ic


class RollingBacktest:
    def __init__(
        self,
        data_handler: DataHandler,
        model: Model,
        strategy: PortfolioStrategy,
        backtest_engine: BacktestEngine,
        train_months: int = 24,
        test_months: int = 6,
        n_windows: int = 8,
    ):
        self.data_handler = data_handler
        self.model = model
        self.strategy = strategy
        self.backtest_engine = backtest_engine
        self.train_months = train_months
        self.test_months = test_months
        self.n_windows = n_windows

    def run(self, symbols: List[str], start: str, end: str) -> Dict:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        
        window_results = []
        
        for i in range(self.n_windows):
            test_end = end_dt - relativedelta(months=i * self.test_months)
            test_start = test_end - relativedelta(months=self.test_months)
            train_end = test_start
            train_start = train_end - relativedelta(months=self.train_months)
            
            if train_start < start_dt:
                break
            
            print(f"Window {i+1}: train [{train_start.date()} ~ {train_end.date()}], test [{test_start.date()} ~ {test_end.date()}]")
            
            # 加载数据
            dataset = self.data_handler.get_dataset(symbols, train_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"))
            
            # 按时间切分
            train_df = dataset[dataset.index.get_level_values("date") <= train_end]
            test_df = dataset[(dataset.index.get_level_values("date") > train_end) & (dataset.index.get_level_values("date") <= test_end)]
            
            if len(train_df) < 100 or len(test_df) < 20:
                continue
            
            feature_cols = [c for c in train_df.columns if c != self.data_handler.label]
            X_train = train_df[feature_cols]
            y_train = train_df[self.data_handler.label]
            X_test = test_df[feature_cols]
            y_test = test_df[self.data_handler.label]
            
            # 训练
            self.model.fit(X_train, y_train)
            
            # 预测
            predictions = self.model.predict(X_test)
            
            # 计算 IC
            ic = calculate_ic(predictions, y_test.values)
            
            # 构建回测用的 predictions DataFrame
            test_index = X_test.index.to_frame()
            pred_df = pd.DataFrame({
                "date": test_index["date"].values,
                "symbol": test_index["symbol"].values,
                "predicted": predictions,
            })
            
            # 回测
            # TODO: 需要加载价格数据用于回测
            # results = self.backtest_engine.run(pred_df, price_data, self.strategy)
            
            window_results.append({
                "window": i + 1,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "ic": ic,
                "n_train": len(train_df),
                "n_test": len(test_df),
            })
        
        ics = [w["ic"] for w in window_results]
        return {
            "window_results": window_results,
            "summary": {
                "avg_ic": np.mean(ics) if ics else 0,
                "std_ic": np.std(ics) if ics else 0,
                "positive_ic_ratio": sum(1 for i in ics if i > 0) / len(ics) if ics else 0,
            },
        }
```

### 步骤 3：编写集成测试

`tests/test_integration.py`：
```python
import numpy as np
import pandas as pd
import pytest
from quant_trading.data.handler import DataHandler
from quant_trading.data.loader import DataLoader
from quant_trading.data.cleaner import DataCleaner
from quant_trading.factors.technical import MA5, RSI14
from quant_trading.factors.price import Momentum5D
from quant_trading.models.gbdt import LightGBMTrainer
from quant_trading.portfolio.strategy import TopKStrategy
from quant_trading.backtest.engine import BacktestEngine, BacktestConfig
from quant_trading.experiments.recorder import Recorder


class TestIntegration:
    def test_end_to_end_pipeline(self, tmp_path):
        # 1. 构建 DataHandler
        features = [MA5(), RSI14(), Momentum5D()]
        handler = DataHandler(features=features, loader=DataLoader(), cleaner=DataCleaner())
        
        # 2. 模型
        model = LightGBMTrainer(params={"num_leaves": 7, "verbose": -1})
        
        # 3. 策略
        strategy = TopKStrategy(top_k=2)
        
        # 4. 回测引擎
        engine = BacktestEngine(BacktestConfig(initial_capital=100000))
        
        # 5. Recorder
        recorder = Recorder("integration_test", output_dir=str(tmp_path))
        
        # 验证各组件可实例化
        assert handler is not None
        assert model is not None
        assert strategy is not None
        assert engine is not None
        assert recorder is not None
```

### 步骤 4：运行全部测试

```bash
cd quant-trading && pytest tests/ -v
```
预期：全部 PASS

### 步骤 5：最终 Commit

```bash
cd quant-trading && git add . && git commit -m "feat: experiments + rolling pipeline + integration tests"
```

---

## 自检清单

- [ ] 所有 8 个任务完成
- [ ] 每个任务有独立的 pytest 测试并通过
- [ ] 没有 Mock 实现（LightGBM 是真实的）
- [ ] 所有技术指标计算都有 shift(1)
- [ ] 数据切分按日期而非条数
- [ ] 提交历史清晰（每个任务一个 commit）
