# quant-trading

A股量化研究框架。当前是一个可运行的 MVP，**长期目标是通过大模型 Skill 实现因子的自动挖掘、自动评估、自动迭代**。

为此，整个架构围绕两个前提设计：
1. **因子必须是可编程的**——不能写死在回测逻辑里，必须能被外部系统（AI）动态注册、替换、评估
2. **实验必须是可记录的**——每次因子改动、模型训练、回测结果，必须能自动保存为结构化数据，供 AI 做上下文学习

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

## 自动化因子研究（Roadmap）

当前 Demo 只有 3 个手动因子（MA、RSI、波动率），回测结果 IR 为负——**这是有意为之**。系统必须先能正确识别"无效因子"，才能谈得上自动发现"有效因子"。

### 为什么当前架构支持 AI 自动化？

**Factor 注册中心的设计意图**

```python
@FactorRegistry.register
class MyFactor(Factor):
    name = "my_factor"
    dependencies = ["close"]
    def compute(self, df):
        return df["close"] * 2
```

当前实现：手写因子 → 手动注册 → 人工评估  
**目标状态**：大模型生成因子代码 → 自动注册 → 自动评估 → 自动判断保留/丢弃

Factor 注册中心提供了 AI 需要的三个 API：
- `FactorRegistry.register(cls)`：动态注册新因子
- `FactorRegistry.list_factors()`：枚举已有因子（避免重复）
- `FactorEvaluator.evaluate(factor, forward_return)`：量化因子的预测能力（IC、Rank IC、IR）

**Recorder 的设计意图**

每次因子实验自动生成目录：
```
experiments/
├── exp_20240602_120000/      # 因子 A
│   ├── params.json            # 因子定义 + 超参
│   ├── metrics.json           # IC / IR / Sharpe
│   └── predictions.csv        # 预测值（可复现）
├── exp_20240602_120015/      # 因子 B
│   └── ...
```

这些结构化数据是大模型做**上下文学习**的素材："上次你试的因子 A 的 IC 是 0.02，因子 B 的 IC 是 -0.01，这次你应该..."

### 大模型 Skill 工作流设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                     大模型 Factor Mining Skill                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: 生成候选因子                                                │
│ 输入：当前因子列表 + 历史实验记录（params + metrics）                  │
│ 输出：Python 代码字符串（compute 函数）                               │
│ 提示词："已有因子 MA/RSI/波动率的 IC 分别为 x/y/z，请生成一个新因子"   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: 沙箱验证                                                    │
│ 输入：生成的代码                                                    │
│ 动作：exec(代码) → 注册 Factor → 运行 DataHandler → 计算 IC          │
│ 输出：IC / Rank IC / IR                                             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: 判断保留/丢弃                                               │
│ 条件：IC > 0.02 且 IR > 0.3 → 保留，进入模型训练                      │
│ 条件：IC < 0 且连续 3 次无改进 → 丢弃，并记录失败原因                  │
│ 输出：实验记录保存到 Recorder                                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: 组合优化                                                    │
│ 输入：保留下来的因子池（如 10 个有效因子）                             │
│ 动作：LightGBM 训练 → 特征重要性排序 → 回测                           │
│ 输出：组合策略的 Sharpe / 最大回撤 / 超额收益                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: 迭代                                                        │
│ 动作：把本轮全部实验记录喂回大模型，进入下一轮因子生成                  │
│ 终止条件：IR 连续 5 轮无提升，或达到最大迭代次数                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键技术挑战

| 挑战 | 当前状态 | 解决方案 |
|------|---------|---------|
| **代码生成安全** | 无 | `exec()` 在受限命名空间中运行，禁止网络/文件系统操作 |
| **因子去重** | 已有 `list_factors()` | 计算因子值的相关系数矩阵，corr > 0.95 视为重复 |
| **过拟合检测** | 已有扩展窗口回测 | 要求 IC 在 7 个 fold 中至少 5 个为正 |
| **计算成本** | 单进程 | 多进程并行评估因子（`multiprocessing.Pool`） |
| **上下文长度** | 无 | 只保留最近 N 次实验记录，摘要化早期历史 |

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
