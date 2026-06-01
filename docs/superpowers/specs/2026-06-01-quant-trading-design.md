# quant-trading 项目设计文档 v1.1

> **面向 AI 代理的工作者：** 本设计基于现有 `rl-quant-trading` 项目的树模型子系统重构而来，参考了微软 Qlib、Zipline Pipeline、Alphalens 的业界最佳实践。

**目标：** 构建一个干净、可测试、模块化的量化交易研究框架，核心能力为「可注册因子 → GBDT 预测 → 策略权重 → 事件驱动回测」。

**架构：** 九层模块化设计（data → factors → models → portfolio → backtest → metrics → experiments → pipeline → utils），每层通过定义良好的协议/接口通信，独立可测。采用 Qlib 的 DataHandler 统一入口、Strategy-Executor 分离、Recorder 实验记录等设计思想。

**技术栈：** Python 3.10+, LightGBM, Pandas, NumPy, pytest

---

## 1. 模块设计

### 1.1 `quant_trading.data` — 数据层

**职责：** 原始数据获取 → 清洗 → 底层数学工具 → DataHandler 统一入口

#### 1.1.1 `data.loader`
```python
class DataLoader:
    def load(self, symbol: str, start: str | None, end: str | None, source: str = "akshare") -> pd.DataFrame:
        """返回统一格式: date(index), open, high, low, close, volume"""
```

#### 1.1.2 `data.cleaner`
```python
class DataCleaner:
    def clean(self, df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """删除停牌/价格异常/流动性极差的日子，返回清洗报告"""
    def get_report(self) -> dict:
        """返回 {original, final, removed, retention, details}"""
```

#### 1.1.3 `data.preprocessing`
底层纯数学工具（无状态，不产因子）：
- `calculate_ma(prices, window, shift=True)`
- `calculate_rsi(prices, window, shift=True)`
- `calculate_macd(prices, shift=True)`
- `calculate_bollinger_bands(prices, shift=True)`
- `rolling_normalize(df, window=252)`
- `time_series_split(df, train_end_date, val_end_date)` — **按日期切分**

#### 1.1.4 `data.handler` — DataHandler 统一入口（Qlib 思想）
```python
class DataHandler:
    def __init__(
        self,
        features: list[Factor],
        label: str = "target_return",
        loader: DataLoader | None = None,
        cleaner: DataCleaner | None = None,
        prediction_horizon: int = 1
    ):
        """配置化定义：用什么因子、预测什么目标"""

    def get_dataset(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """
        输出: MultiIndex (date, symbol), columns = [factor_1, ..., factor_n, target]
        流程: load → clean → compute factors → create target → rolling_normalize → dropna
        """
```

**设计约束：**
- 所有技术指标计算必须 `shift(1)`，防止前瞻偏差
- 标准化必须使用滚动窗口（默认 252 日），禁止全局统计
- 切分必须按时间顺序，禁止随机切分

---

### 1.2 `quant_trading.factors` — 因子层（Zipline Pipeline + Alphalens 思想）

**职责：** 因子的定义、注册、计算、评估

#### 1.2.1 `factors.base`
```python
class Factor(ABC):
    name: str
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """输入清洗后的 ohlcv，输出因子序列（已 shift(1)）"""

FACTOR_REGISTRY: dict[str, type[Factor]] = {}

def register_factor(cls: type[Factor]) -> type[Factor]: ...
```

#### 1.2.2 `factors.technical` / `factors.price` / `factors.volume`
技术因子、价格因子、成交量因子的具体实现。全部通过 `@register_factor` 注册。

#### 1.2.3 `factors.evaluator` — Alphalens 式评估
```python
class FactorEvaluator:
    def evaluate(self, factor: Factor, df: pd.DataFrame, forward_returns: pd.Series) -> dict:
        """
        返回:
        {
            "ic": float,           # 信息系数 (Pearson)
            "rank_ic": float,      # Rank IC (Spearman)
            "ic_std": float,       # IC 标准差
            "ir": float,           # 信息比率 = IC_mean / IC_std
            "turnover": float,     # 因子换手率
            "quantile_returns": pd.DataFrame,  # 分位数收益
            "half_life": float     # 因子衰减半衰期
        }
        """
```

---

### 1.3 `quant_trading.models` — 模型层

**职责：** 树模型训练、交叉验证、过拟合检测

#### 1.3.1 `models.base` — Model 协议（极薄一层）
```python
class Model(ABC):
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: tuple | None = None) -> "Model": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "Model": ...
```

#### 1.3.2 `models.gbdt`
```python
class LightGBMTrainer(Model):
    def __init__(self, params: dict | None = None): ...
    def feature_importance(self) -> dict[str, float]: ...
```

#### 1.3.3 `models.cv`
```python
class TimeSeriesCV(BaseCrossValidator): ...
class WalkForwardCV(BaseCrossValidator): ...
```

#### 1.3.4 `models.validation`
```python
def detect_overfitting(train_ic: float, val_ic: float, threshold: float = 0.02) -> tuple[bool, float]: ...
def analyze_learning_curve(train_scores: list, val_scores: list) -> dict: ...
```

**设计约束：**
- `LightGBMTrainer` 必须是真实实现，基于 `lightgbm` 库，禁止 Mock
- 交叉验证必须保持时间顺序

---

### 1.4 `quant_trading.portfolio` — 规划层（Qlib Strategy 思想）

**职责：** 根据模型预测生成目标持仓权重

#### 1.4.1 `portfolio.base`
```python
class PortfolioStrategy(ABC):
    def generate_weights(
        self,
        predictions: pd.DataFrame,  # columns: date, symbol, predicted
        current_portfolio: dict[str, float] | None = None
    ) -> pd.Series:
        """返回: {symbol: target_weight}"""
```

#### 1.4.2 `portfolio.strategy`
```python
class TopKStrategy(PortfolioStrategy):
    def __init__(self, top_k: int = 5, weight_method: str = "equal"): ...  # "equal" | "confidence"

class LongShortStrategy(PortfolioStrategy): ...
class MarketNeutralStrategy(PortfolioStrategy): ...
```

---

### 1.5 `quant_trading.backtest` — 回测层

**职责：** 模拟交易执行、绩效计算

#### 1.5.1 `backtest.executor` — Executor（Qlib 思想）
```python
class Executor:
    def execute(
        self,
        target_weights: pd.Series,
        current_positions: dict[str, dict],
        price_data: dict[str, pd.DataFrame],
        date: datetime,
        config: BacktestConfig
    ) -> list[Trade]:
        """
        模拟撮合，返回实际成交的 trades。
        考虑：T+1、手续费、滑点、停牌、整手交易
        """
```

#### 1.5.2 `backtest.engine` — 事件驱动回测引擎
```python
@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000
    commission_rate: float = 0.0003
    slippage: float = 0.001
    position_size: float = 1.0
    max_positions: int = 10
    stop_loss: float | None = 0.05
    take_profit: float | None = 0.10

class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None): ...

    def run(
        self,
        predictions: pd.DataFrame,
        price_data: dict[str, pd.DataFrame],
        strategy: PortfolioStrategy,
        rebalance_freq: str = "daily"
    ) -> dict:
        """运行回测，返回绩效指标"""
```

**设计约束：**
- T+1 执行：信号日收盘后决策，次日开盘价成交
- 手续费万3，滑点 0.1%
- 停牌日跳过交易，持仓市值用最后已知价格
- A股整手交易（100股）

---

### 1.6 `quant_trading.metrics` — 指标层

**职责：** 金融指标计算

```python
def calculate_ic(predictions: np.ndarray, returns: np.ndarray) -> float: ...
def calculate_rank_ic(predictions: np.ndarray, returns: np.ndarray) -> float: ...
def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.03) -> float: ...
def calculate_max_drawdown(prices: pd.Series) -> float: ...
def calculate_calmar_ratio(returns: pd.Series, prices: pd.Series) -> float: ...
def calculate_excess_return(strategy_return: float, benchmark_return: float) -> float: ...
```

**设计约束：**
- 超额收益使用对数收益率计算（复利正确）
- 所有计算必须有除零保护

---

### 1.7 `quant_trading.experiments` — 实验追踪（Qlib Recorder 思想）

**职责：** 一次实验的所有产物自动归档

```python
class Recorder:
    def __init__(self, experiment_name: str, output_dir: str = "experiments"): ...
    def log_model(self, model: Model, path: str): ...
    def log_predictions(self, predictions: pd.DataFrame): ...
    def log_backtest_report(self, report: dict): ...
    def log_config(self, config: dict): ...
    def get_artifact(self, name: str) -> Any: ...
```

归档结构：
```
experiments/
└── 20260601_120000_exp_001/
    ├── config.yaml
    ├── model.pkl
    ├── predictions.csv
    ├── backtest_report.json
    └── feature_importance.csv
```

---

### 1.8 `quant_trading.pipeline` — 研究流水线

**职责：** Walk-forward 多窗口评估（核心差异化能力）

```python
class RollingBacktest:
    def __init__(
        self,
        data_handler: DataHandler,
        model: Model,
        strategy: PortfolioStrategy,
        backtest_engine: BacktestEngine,
        train_months: int = 24,
        test_months: int = 6,
        n_windows: int = 8
    ): ...

    def run(self, symbols: list[str], start: str, end: str) -> dict:
        """
        对每个窗口:
            1. 获取训练/测试数据
            2. 训练模型
            3. 预测
            4. 回测
        返回: 多窗口汇总指标 {avg_ic, avg_return, avg_sharpe, consistency}
        """
```

---

### 1.9 `quant_trading.utils` — 工具层

```python
# utils/config.py
class Config:
    def __init__(self, config_dict: dict): ...
    def get(self, key: str, default: Any = None) -> Any: ...  # 支持 "model.max_depth"
    def to_dict(self) -> dict: ...

def load_config(path: str) -> Config: ...
def save_config_snapshot(config: Config, log_dir: str) -> None: ...
```

---

## 2. 数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                        研究流水线 (RollingBacktest)                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────┐     ┌─────────────────┐     ┌─────────────────────────┐
│  DataLoader │────▶│  DataCleaner    │────▶│   DataHandler           │
│ (AKShare/   │     │ (停牌/异常价格/  │     │   (因子计算 + 目标变量   │
│  腾讯财经)   │     │  流动性过滤)     │     │    + 滚动标准化)         │
└─────────────┘     └─────────────────┘     └───────────┬─────────────┘
                                                        │
                           ┌────────────────────────────┘
                           ▼
              ┌────────────────────────┐
              │  factors/*.py          │
              │  (Factor.compute())    │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  Model.fit/predict     │
              │  (LightGBM)            │
              └───────────┬────────────┘
                          │ predictions
                          ▼
              ┌────────────────────────┐
              │  PortfolioStrategy     │
              │  (generate_weights)    │
              └───────────┬────────────┘
                          │ target_weights
                          ▼
              ┌────────────────────────┐
              │  Executor              │
              │  (模拟撮合: T+1/手续费/滑点)│
              └───────────┬────────────┘
                          │ trades
                          ▼
              ┌────────────────────────┐
              │  BacktestEngine        │
              │  (持仓/账户/绩效计算)   │
              └───────────┬────────────┘
                          │ metrics
                          ▼
              ┌────────────────────────┐
              │  Recorder              │
              │  (归档: model+pred+report)│
              └────────────────────────┘
```

---

## 3. 与旧项目的关键差异

| 项目 | 旧 `rl-quant-trading/src/` | 新 `quant-trading` |
|------|---------------------------|-------------------|
| **因子系统** | 硬编码在 `StockDataLoader._create_features()` | **可注册 Factor + Registry + Evaluator** |
| **因子评估** | 无 | **Alphalens 式 IC/IR/换手率/衰减分析** |
| **数据切分** | 按总条数 `iloc[:train_end]` | **按日期切分** |
| **LightGBM** | `MockLGBMModel`（假模型） | **真实 `lightgbm`** |
| **策略与回测** | 混在 `BacktestEngine.run()` 里 | **Strategy → Executor 分离（Qlib 式）** |
| **实验追踪** | 简单的 `ExperimentLogger` | **Recorder 结构化归档（模型+预测+报告）** |
| **配置管理** | 无 | **YAML + 点号访问 + 快照** |
| **数据清洗** | 有但散落 | **独立 DataCleaner + 质量报告** |

---

## 4. 测试策略

每层独立测试，使用 pytest：

| 测试模块 | 覆盖内容 |
|---------|---------|
| `test_data/` | 特征计算 shift(1) 验证、滚动标准化、日期切分、数据清洗 |
| `test_factors/` | 因子注册、因子计算、单因子 IC/IR/换手率评估 |
| `test_models/` | LightGBM fit/predict（小数据）、CV 顺序验证、过拟合检测 |
| `test_portfolio/` | 目标权重生成正确性（等权/confidence） |
| `test_backtest/` | T+1 执行、手续费计算、滑点方向、停牌处理 |
| `test_metrics/` | IC/夏普/回撤计算正确性、除零保护 |
| `test_experiments/` | Recorder 归档结构正确 |
| `test_pipeline/` | Walk-forward 窗口切分、多窗口汇总 |

**TDD 铁律：** 先写测试 → 运行确认失败 → 写最少实现 → 运行确认通过 → 重构。

---

## 5. 范围排除（YAGNI）

以下功能**不在本次重构范围内**：
- RL / 模仿学习（保留在旧项目）
- LSTM / Transformer 实现（保留 Model 协议，便于后续接入）
- Web UI / Streamlit / CLI
- 实时交易接口（execution/interface.py 已删除）
- 向量化回测（暂缓，只保留事件驱动）
- 数据库（Postgres/MongoDB）
- Docker / 部署脚本
- Expression DSL（Qlib 的 `$close / Ref($close, 5)` 语法）
- Mod 插件系统（RQAlpha 式）

---

## 6. 规格自检

- [x] 占位符扫描：无 "TODO"、"待定"、"后续实现"
- [x] 内部一致性：模块接口与数据流一致
- [x] 范围检查：聚焦因子+GBDT+回测，可被一个实现计划覆盖
- [x] 模糊性检查：切分方式、shift(1)、T+1、Strategy/Executor 边界均已明确
- [x] 业界对齐：DataHandler、Strategy-Executor 分离、Recorder、Factor Registry 均对标 Qlib/Zipline/Alphalens
