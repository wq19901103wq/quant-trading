# quant-trading

A股量化研究框架，设计目标：**用工程化的方式解决量化研究中的可重复性问题**。

核心挑战不是"跑出一个回测曲线"，而是让每一步（数据加载 → 因子计算 → 模型训练 → 策略回测 → 指标评估）都可追溯、可复现、可增量迭代。为此，整个系统围绕**数据血缘**和**时间边界严格性**两个原则构建。

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                         数据层 (Data Layer)                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐  │
│  │ DataSource│ → │ DataCleaner│ → │ FactorComputer│ → │ DataHandler │  │
│  │ (AKShare) │   │ (异常过滤)  │   │ (groupby计算) │   │ (MultiIndex) │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        模型层 (Model Layer)                        │
│  ┌──────────────┐   ┌────────────────┐   ┌──────────────────┐  │
│  │ LightGBM     │   │ TimeSeriesCV   │   │ evaluate_regression│  │
│  │ (GBDT回归)    │   │ (扩展窗口切分)  │   │ (MSE/RMSE/R²/IC)  │  │
│  └──────────────┘   └────────────────┘   └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       策略层 (Strategy Layer)                      │
│  ┌─────────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │ PortfolioStrategy│ → │ Executor     │ → │ BacktestEngine  │  │
│  │ (TopK/Rank/LS)  │   │ (佣金/滑点)   │   │ (日频再平衡)     │  │
│  └─────────────────┘   └──────────────┘   └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        指标层 (Metrics Layer)                      │
│  ┌──────────────────┐   ┌─────────────────────────────────────┐  │
│  │ calculate_metrics │   │ calculate_excess_metrics             │  │
│  │ (Sharpe/Drawdown) │   │ (IR/Tracking Error/Beta/相对回撤)    │  │
│  └──────────────────┘   └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 关键技术决策

### 1. 为什么用 GBDT（LightGBM）而非深度学习？

| 维度 | GBDT | 深度学习 |
|------|------|---------|
| **特征交互** | 天然处理异构特征（价格、成交量、技术指标），自动学习非线性组合 | 需要大量工程化构造输入结构 |
| **数据量** | 500只股票 × 10年 ≈ 125万条，GBDT 在此量级表现优异 | 需要千万级样本才能拉开差距 |
| **可解释性** | `feature_importance()` 直接输出增益，便于因子迭代 | 黑盒，难以归因 |
| **训练速度** | 秒级训练，支持快速 walk-forward 验证 | 分钟级起步，迭代成本高 |
| **过拟合控制** | `feature_fraction` + `bagging_fraction` + 叶子数限制 | 需要复杂的正则化和早停策略 |

**结论**：在 A 股日频截面预测场景下，GBDT 是**性价比最高的基线模型**。深度学习留给更高频（分钟级）或更多模态数据（文本、图像）的场景。

### 2. 为什么是扩展窗口（Expanding Window）而非滑动窗口？

量化回测中常见的三种窗口设计：

| 类型 | 训练数据 | 测试数据 | 适用场景 | 缺陷 |
|------|---------|---------|---------|------|
| **固定窗口** | 2015-2017（固定3年） | 2018, 2019, ... | 数据分布稳定的环境 | 丢弃了大量历史信息，后期 fold 训练数据过少 |
| **滑动窗口** | 2016-2018, 2017-2019, ... | 每年1年 | 数据分布随时间漂移 | 各 fold 训练量相同但不独立，统计检验困难 |
| **扩展窗口** ✅ | 2015-2017, 2015-2018, ... | 每年1年 | 真实交易场景（用全部历史训练） | 后期 fold 训练数据更多，但不公平性可被接受 |

**选择扩展窗口的核心原因**：
- 贴近真实交易流程——交易员不会故意遗忘 3 年前的数据
- 训练量递增，模型能力随时间增强，这是**合理的
- 7 个 fold 覆盖 2018-2024，测试窗口**完全不重叠**，避免了信息泄漏

### 3. 为什么是 MultiIndex `(date, symbol)`？

单股票回测用 `date` 做 index 就够了，但多股票场景下会暴露三个问题：

**问题1：Index 重复**
```python
# 错误：3 只股票的 2020-01-02 会生成 3 个相同的 index
df = pd.concat([df_a, df_b, df_c])  # index 重复
```

**问题2：跨股票污染**
```python
# 危险：shift(-1) 会把股票A的明天收益和股票B的明天收益混在一起
df["return_1d"] = df["close"].pct_change().shift(-1)  # 跨股票！
```

**问题3：截面操作困难**
```python
# 需要每天对所有股票做排序，单 index 无法表达"同一日期不同股票"
```

**MultiIndex 解决方案**：
```python
df.set_index(["date", "symbol"], inplace=True)

# 按股票分组计算，彻底隔离
df["ma_20"] = df.groupby(level="symbol")["close"].transform(
    lambda x: x.rolling(20).mean().shift(1)
)

# 截面 pivot：日期 × 股票
df["pred"].unstack(level="symbol")  # 每行是一个截面
```

### 4. 为什么用等权组合作为基准而非沪深300？

| 基准类型 | 优点 | 缺陷 |
|---------|------|------|
| **沪深300** | 市场公认，易于横向对比 | AKShare 指数接口不稳定（实践中被限流）；市值加权与策略持仓结构差异大 |
| **等权组合** ✅ | 直接从已有股票数据构造，零外部依赖；与 TopK 策略的"选股"逻辑直接可比 | 不是市场基准，无法与公募基金对比 |

**等权基准的计算**：
```python
# 每日所有股票的等权收益
benchmark = df["close"].groupby(level="symbol").pct_change()
benchmark = benchmark.groupby(level=0).mean()  # 按日期等权平均
```

**信息比率 IR = 策略超额收益 / 跟踪误差**，直接回答："相对于随机选股，你的策略有多强？"

### 5. 为什么设计 Factor 注册中心？

硬编码因子的经典问题：
```python
# 反模式：因子散落在各个文件里，无法统一管理
df["ma_20"] = df["close"].rolling(20).mean()
df["rsi_14"] = ...  # 另一个文件里又写了一遍
```

**注册中心的优势**：
- **依赖声明**：`dependencies = ["close"]`，DataHandler 自动检查列是否存在
- **可枚举**：`FactorRegistry.list_factors()` 一键查看全部因子
- **可扩展**：新增因子只需一个 `@FactorRegistry.register` 装饰器
- **可评估**：`FactorEvaluator` 统一计算 IC、Rank IC、IR，支持因子筛选

---

## 数据流设计

从原始数据到回测结果，每步的数据格式转换：

| 步骤 | 输入 | 输出 | 关键操作 |
|------|------|------|---------|
| **DataSource.load()** | 股票代码 + 日期范围 | `DataFrame(date, open, high, low, close, volume)` | AKShare 新浪接口，中文列名映射 |
| **DataCleaner.clean()** | 原始 OHLCV | 清洗后的 OHLCV | 负价格过滤、价格关系异常检测、涨跌停识别、零成交量剔除 |
| **FactorComputer** | 清洗后数据 | `DataFrame` + 因子列 | `groupby(level="symbol")` 计算，**shift(1) 防 lookahead** |
| **DataHandler** | 多股票因子数据 | `MultiIndex DataFrame(date, symbol)` | 按日期切分 train/test，NA 处理（drop/ffill/zero） |
| **Model.fit()** | `X: (n_samples, n_features), y: (n_samples,)` | 训练好的 Booster | LightGBM，支持 `eval_set` + 早停 |
| **Model.predict()** | `X_test` | `(n_test,)` 预测值 | 直接预测未来收益值（回归） |
| **unstack()** | 长格式预测 `(date, symbol) → pred` | 截面矩阵 `pred_pivot(date × symbol)` | 每行是一个交易日，每列是一只股票 |
| **PortfolioStrategy.generate_weights()** | 截面预测 `Series(symbol → pred)` | `Series(symbol → weight)` | TopK：选预测最高的 K 只，等权分配 |
| **Executor.execute()** | 目标权重 + 当前权重 + 价格 | 执行结果（佣金、滑点、实际权重） | 佣金 = trade_value × rate，min_commission 保底 |
| **BacktestEngine.run()** | 每日权重矩阵 + 价格矩阵 | 收益序列 + 净值曲线 + 交易历史 | 日频再平衡，次日收益结算 |
| **calculate_metrics()** | 日收益序列 | Sharpe / 最大回撤 / Calmar | 年化因子 √252 |
| **calculate_excess_metrics()** | 策略收益 + 基准收益 | IR / 跟踪误差 / Beta / 相对回撤 | 日超额收益的统计特征 |

---

## 工程实践

### TDD（测试驱动开发）

57 个单元测试覆盖全部核心模块，包括：
- **Lookahead 防护**：`test_ma_shift_prevents_lookahead` 验证 MA 是否 shift(1)
- **数据清洗边界**：`test_remove_zero_volume`、`test_remove_price_relation_anomaly`
- **回测正确性**：`test_no_trade_no_cost`（零交易零佣金）、`test_trade_generates_commission`
- **模型协议**：`test_predict_before_fit_raises`（未训练抛异常）、`test_save_and_load`

### 错误隔离

```python
# 单只股票加载失败不中断全量
for sym in symbols:
    try:
        raw = loader.load(sym, start_date, end_date)
    except Exception:
        logger.warning("跳过 %s", sym)  # 继续下一只
```

500 只股票中 22 只加载失败，剩余 478 只正常参与回测。

### 本地 CSV 缓存

```python
class CachedAKShareDataSource:
    def load(self, symbol, start, end):
        if cache_exists:
            return pd.read_csv(cache_path)  # 秒级
        df = akshare.load(...)            # 网络请求
        df.to_csv(cache_path)              # 持久化
        time.sleep(1)                      # throttle 防限流
```

首次 500 只 × 10 年 ≈ 10-15 分钟，后续秒级运行。

---

## 量化专业深度

### 截面模型 vs 时序模型

本项目是**截面模型**（Cross-sectional）：
- 每天对所有股票做预测，比较的是"今天哪只股票更好"
- 不关心单只股票的绝对收益，关心的是**相对排名**
- 回测策略是 TopK 截面等权，而非单股票择时

时序模型（Time-series）会预测"明天大盘涨还是跌"，适合 ETF 择时；截面模型适合 Alpha 选股。

### Lookahead 防护

量化中最隐蔽的 bug 是未来信息泄漏。本项目的防护措施：

| 场景 | 风险 | 防护 |
|------|------|------|
| 移动平均 | `rolling(20).mean()` 包含当日 | `rolling(20).mean().shift(1)` 只用前 19 日 |
| 收益标签 | `pct_change()` 用当日收盘价 | `pct_change().shift(-1)` 预测的是次日收益 |
| 滚动归一化 | `rolling(252).mean()` 包含未来 | `rolling(252).mean().shift(1)` |
| 截面操作 | `unstack()` 后按行排序 | 每行独立处理，不跨日期泄漏 |

### 信息比率 IR 的解读

```
IR > 0.5   → 策略有显著 Alpha，可考虑实盘
IR 0.2-0.5 → 弱 Alpha，需因子增强
IR < 0     → 跑输基准，策略无效
```

当前 Demo（3 个简单因子）的 IR ≈ -0.31，说明因子组合没有选股能力。这**证明了系统能正确检测无效策略**——如果随便跑个回测都是正 IR，那指标本身就有问题。

---

## 快速开始

```bash
# 安装
conda create -n quant-trading python=3.10
conda activate quant-trading
pip install pandas numpy lightgbm scikit-learn scipy pyyaml pytest akshare

# 运行完整流水线（500 只 × 10 年扩展窗口回测）
python examples/run_real_data.py

# 测试
pytest tests/ -v
```

---

## 项目结构

```
quant-trading/
├── src/                    # 核心源码
│   ├── data/               # loader, cleaner, preprocessing, handler
│   ├── factors/            # base, registry, technical/price/volume, evaluator
│   ├── models/             # base, gbdt, cross_validation, validation
│   ├── portfolio/          # TopK, RankWeighted, LongShort strategies
│   ├── backtest/           # executor, engine
│   ├── metrics/            # absolute + excess metrics
│   ├── pipeline/           # RollingBacktest
│   ├── experiments/        # Recorder
│   └── utils/              # Config
├── tests/                  # 57 单元测试
├── examples/               # 端到端 Demo
├── docs/                   # 设计文档
├── pyproject.toml
└── README.md
```

---

## 技术栈

- **Python 3.10** | **pandas 2.3** | **numpy 2.1**
- **LightGBM 4.6**（GBDT 回归）
- **scikit-learn**（验证指标）
- **AKShare**（A 股数据源，新浪接口）
- **pytest**（TDD）

## License

MIT
