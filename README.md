# quant-trading

A股量化交易研究框架，支持从数据获取到回测的完整流水线。

## 特性

- **数据源**：AKShare（新浪接口）+ 本地 CSV 缓存，避免重复网络请求
- **因子系统**：可插拔因子注册中心，内置技术指标/价格/成交量因子
- **模型**：LightGBM GBDT + 时序交叉验证
- **回测**：截面多股票回测 + 扩展窗口滚动验证
- **指标**：绝对收益（Sharpe、最大回撤）+ 超额收益（IR、跟踪误差、Beta）
- **实验记录**：自动保存参数、指标、预测值、净值曲线

## 安装

```bash
# 创建 conda 环境
conda create -n quant-trading python=3.10
conda activate quant-trading

# 安装依赖
pip install pandas numpy lightgbm scikit-learn scipy pyyaml pytest akshare

# 验证
pytest tests/
```

## 快速开始

### 1. 运行端到端 Demo（500 只股票 × 10 年扩展窗口回测）

```bash
conda activate quant-trading
python examples/run_real_data.py
```

首次运行约 10-15 分钟（网络拉取 500 只股票数据），后续从缓存秒级运行。

### 2. 自定义流水线

```python
from quant_trading.data.loader import CachedAKShareDataSource, DataLoader
from quant_trading.data.handler import DataHandler
from quant_trading.models.gbdt import LightGBMModel
from quant_trading.portfolio.strategy import TopKStrategy
from quant_trading.backtest.executor import Executor
from quant_trading.backtest.engine import BacktestEngine

# 加载数据
source = CachedAKShareDataSource(cache_dir="./data/cache", provider="sina")
loader = DataLoader(source)

handler = DataHandler(
    data_loader=loader,
    symbols=["000001", "000002", "600036"],
    start_date="2020-01-01",
    end_date="2024-12-31",
    features=["ma_20", "rsi_14", "volatility_20d"],
    label="return_5d",
)

# 训练模型
model = LightGBMModel()
model.fit(handler.get_feature_matrix(), handler.get_label_series())

# 回测
strategy = TopKStrategy(k=10)
executor = Executor(commission_rate=0.001)
engine = BacktestEngine(strategy, executor, initial_capital=10_000_000)
result = engine.run(predictions_df, prices_df)
print(result["metrics"])
```

## 项目结构

```
quant-trading/
├── src/quant_trading/
│   ├── data/           # loader, cleaner, preprocessing, handler
│   ├── factors/        # base, registry, technical/price/volume, evaluator
│   ├── models/         # base, gbdt, cross_validation, validation
│   ├── portfolio/      # TopK, RankWeighted, LongShort strategies
│   ├── backtest/       # executor, engine
│   ├── metrics/        # absolute + excess metrics
│   ├── pipeline/       # RollingBacktest
│   ├── experiments/    # Recorder
│   └── utils/          # Config
├── tests/              # 57+ 单元测试
├── examples/           # 端到端 Demo
└── docs/               # 设计文档
```

## 核心模块说明

### DataHandler
统一数据入口，支持 MultiIndex `(date, symbol)`，自动处理多股票因子计算（按 `groupby(level="symbol")` 避免跨股票污染）。

### Factor Registry
```python
from quant_trading.factors.registry import FactorRegistry
from quant_trading.factors.base import Factor

@FactorRegistry.register
class MyFactor(Factor):
    name = "my_factor"
    dependencies = ["close"]
    def compute(self, df):
        return df["close"] * 2
```

### 滚动回测（扩展窗口）
```python
# 训练窗口逐年扩展：2015-2017 → 2015-2018 → ... → 2015-2023
# 测试窗口：2018, 2019, ..., 2024
```

### 超额收益指标
- `excess_return`: 组合收益 - 基准收益
- `information_ratio`: 日均超额 / 跟踪误差
- `tracking_error`: 超额收益的年化标准差
- `beta`: 组合对基准的敏感度

## 测试

```bash
pytest tests/ -v
```

57 个单元测试覆盖全部核心模块。

## 技术栈

- Python 3.10
- pandas 2.3, numpy 2.1
- LightGBM 4.6
- scikit-learn
- AKShare（A 股数据源）
- pytest

## License

MIT
