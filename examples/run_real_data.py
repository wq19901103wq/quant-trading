#!/usr/bin/env python3
"""端到端 Demo：用 AKShare 真实 A 股数据跑完整量化流水线.

用法:
    cd ~/quant-trading
    conda run -n quant-trading python examples/run_real_data.py
"""
import sys
from pathlib import Path

# 将 src 加入路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import logging
import pandas as pd
import numpy as np

from quant_trading.data.loader import CachedAKShareDataSource, DataLoader
from quant_trading.data.cleaner import DataCleaner
from quant_trading.data.handler import DataHandler
from quant_trading.data.preprocessing import calculate_ma, calculate_rsi
from quant_trading.models.gbdt import LightGBMModel
from quant_trading.portfolio.strategy import TopKStrategy
from quant_trading.backtest.executor import Executor
from quant_trading.backtest.engine import BacktestEngine
from quant_trading.experiments.recorder import Recorder
from quant_trading.metrics.core import calculate_metrics
from quant_trading.models.validation import evaluate_regression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo")

# =============================================================================
# 配置参数
# =============================================================================
SYMBOLS = ["000001", "000002", "600036"]  # 平安银行、万科A、招商银行
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
TRAIN_END = "2023-12-31"
CACHE_DIR = "./data/cache"
EXPERIMENT_DIR = "./experiments"


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """为单只股票计算因子（避免 lookahead）."""
    df = df.copy()
    df["ma_20"] = calculate_ma(df["close"], window=20, shift=True)
    df["rsi_14"] = calculate_rsi(df["close"], window=14, shift=True)
    df["return_1d"] = df["close"].pct_change().shift(-1)  # 标签：次日收益
    df["volatility_20d"] = df["close"].pct_change().rolling(20).std().shift(1)
    return df


def main():
    logger.info("=" * 60)
    logger.info("Quant Trading Pipeline Demo — Real Data")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # 1. 数据加载（带本地缓存）
    # -------------------------------------------------------------------------
    logger.info("[1/6] 从 AKShare 加载数据...")
    source = CachedAKShareDataSource(cache_dir=CACHE_DIR, adjust="qfq", provider="sina")
    loader = DataLoader(source)

    all_frames = []
    for sym in SYMBOLS:
        try:
            raw = loader.load(sym, START_DATE, END_DATE, use_cache=True)
            logger.info("  %s: %d 条原始数据", sym, len(raw))
            cleaner = DataCleaner()
            cleaned = cleaner.clean(raw, symbol=sym)
            logger.info("  %s: 清洗后 %d 条 (保留率 %.1f%%)",
                        sym, len(cleaned), cleaner.get_report()["retention"] * 100)
            factored = compute_factors(cleaned)
            factored["symbol"] = sym
            all_frames.append(factored)
        except Exception as e:
            logger.error("加载 %s 失败: %s", sym, e)

    if len(all_frames) < len(SYMBOLS):
        logger.warning("部分股票加载失败，仅使用 %d / %d 只", len(all_frames), len(SYMBOLS))

    combined_df = pd.concat(all_frames)
    combined_df = combined_df.dropna(subset=["ma_20", "rsi_14", "volatility_20d", "return_1d"])
    logger.info("合并后共 %d 条样本", len(combined_df))

    # -------------------------------------------------------------------------
    # 2. 时序拆分训练 / 测试
    # -------------------------------------------------------------------------
    logger.info("[2/6] 时序拆分训练集 / 测试集...")
    train_df = combined_df[combined_df.index <= TRAIN_END]
    test_df = combined_df[combined_df.index > TRAIN_END]
    logger.info("训练集: %d 条, 测试集: %d 条", len(train_df), len(test_df))

    feature_cols = ["ma_20", "rsi_14", "volatility_20d"]
    label_col = "return_1d"

    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    # -------------------------------------------------------------------------
    # 3. 模型训练
    # -------------------------------------------------------------------------
    logger.info("[3/6] 训练 LightGBM 模型...")
    model = LightGBMModel(params={
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    })
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    logger.info("预测完成: %d 条预测值", len(preds))

    # 验证指标
    val_metrics = evaluate_regression(y_test, preds)
    logger.info("验证指标 — MSE: %.6f, RMSE: %.6f, R²: %.4f",
                val_metrics["mse"], val_metrics["rmse"], val_metrics["r2"])

    # 特征重要性
    importance = model.feature_importance()
    logger.info("特征重要性:")
    for feat, imp in importance.items():
        logger.info("  %s: %.2f", feat, imp)

    # -------------------------------------------------------------------------
    # 4. 回测（简化：每日按预测值排序，Top-1 等权持有）
    # -------------------------------------------------------------------------
    logger.info("[4/6] 回测模拟...")
    test_df = test_df.copy()
    test_df["pred"] = preds

    # 构造每日截面预测 DataFrame（日期 x 股票）
    pred_pivot = test_df.pivot(columns="symbol", values="pred")
    price_pivot = test_df.pivot(columns="symbol", values="close")

    # 只保留有完整数据的日期
    common_dates = pred_pivot.index.intersection(price_pivot.index)
    pred_pivot = pred_pivot.loc[common_dates].ffill()
    price_pivot = price_pivot.loc[common_dates].ffill()

    engine = BacktestEngine(
        strategy=TopKStrategy(k=1),
        executor=Executor(commission_rate=0.001, min_commission=0),
        initial_capital=1_000_000,
    )
    result = engine.run(pred_pivot, price_pivot)
    metrics = result["metrics"]
    logger.info("回测完成!")
    logger.info("  总收益:     %.2f%%", metrics["total_return"] * 100)
    logger.info("  年化收益:   %.2f%%", metrics["annualized_return"] * 100)
    logger.info("  波动率:     %.2f%%", metrics["volatility"] * 100)
    logger.info("  Sharpe:     %.3f", metrics["sharpe_ratio"])
    logger.info("  Sortino:    %.3f", metrics["sortino_ratio"])
    logger.info("  最大回撤:   %.2f%%", metrics["max_drawdown"] * 100)
    logger.info("  Calmar:     %.3f", metrics["calmar_ratio"])

    # -------------------------------------------------------------------------
    # 5. 记录实验结果
    # -------------------------------------------------------------------------
    logger.info("[5/6] 保存实验记录...")
    recorder = Recorder(EXPERIMENT_DIR, experiment_name="real_data_demo")
    recorder.log_params({
        "symbols": SYMBOLS,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "train_end": TRAIN_END,
        "features": feature_cols,
        "label": label_col,
        "model": "LightGBM",
    })
    recorder.log_metrics({**val_metrics, **metrics})
    recorder.log_artifact("predictions.csv", pd.Series(preds, index=y_test.index, name="prediction"))
    recorder.log_artifact("portfolio_values.csv", result["portfolio_values"])
    logger.info("结果已保存到: %s", recorder.get_run_dir())

    # -------------------------------------------------------------------------
    # 6. 每日收益摘要
    # -------------------------------------------------------------------------
    logger.info("[6/6] 收益摘要:")
    returns = result["returns"]
    logger.info("  日均收益:   %.4f%%", returns.mean() * 100)
    logger.info("  收益标准差: %.4f%%", returns.std() * 100)
    logger.info("  正收益天数: %d / %d (%.1f%%)",
                (returns > 0).sum(), len(returns), (returns > 0).mean() * 100)

    logger.info("=" * 60)
    logger.info("Demo 完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
