#!/usr/bin/env python3
"""端到端 Demo：用 AKShare 真实 A 股数据跑完整量化流水线（大规模版本）.

默认配置：500 只股票 × 10 年数据
首次运行约 8-10 分钟（网络拉取），后续从缓存秒级运行.

用法:
    cd ~/quant-trading
    conda run -n quant-trading python examples/run_real_data.py
"""
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import logging
import time
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
from quant_trading.models.validation import evaluate_regression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo")

# =============================================================================
# 配置参数
# =============================================================================
NUM_STOCKS = 500          # 股票数量
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
TRAIN_END = "2022-12-31"
CACHE_DIR = "./data/cache"
EXPERIMENT_DIR = "./experiments"


def get_stock_list(n: int = 500) -> list:
    """获取A股列表，过滤ST股，返回前n只."""
    import akshare as ak
    df = ak.stock_info_a_code_name()
    # 过滤ST/*ST/退市
    df = df[~df["name"].str.contains(r"ST|退|摘", na=False, regex=True)]
    # 过滤B股（代码以2/9开头）和非标准代码
    df = df[df["code"].str.match(r"^[06\d]\d{5}$")]
    symbols = df["code"].head(n).tolist()
    logger.info("Selected %d stocks", len(symbols))
    return symbols


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """为每只股票独立计算因子（避免跨股票污染）."""
    df = df.copy()
    # 按 symbol 分组计算
    df["ma_20"] = df.groupby(level="symbol")["close"].transform(
        lambda x: calculate_ma(x, window=20, shift=True)
    )
    df["rsi_14"] = df.groupby(level="symbol")["close"].transform(
        lambda x: calculate_rsi(x, window=14, shift=True)
    )
    df["volatility_20d"] = df.groupby(level="symbol")["close"].transform(
        lambda x: x.pct_change().rolling(20).std().shift(1)
    )
    # 标签：未来5日收益（比1日更容易预测）
    df["return_5d"] = df.groupby(level="symbol")["close"].transform(
        lambda x: x.pct_change(5).shift(-5)
    )
    return df


def main():
    logger.info("=" * 70)
    logger.info("Quant Trading Pipeline Demo — Large Scale")
    logger.info("Config: %d stocks × %s ~ %s", NUM_STOCKS, START_DATE, END_DATE)
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # 1. 获取股票列表
    # -------------------------------------------------------------------------
    logger.info("[1/7] 获取股票列表...")
    symbols = get_stock_list(NUM_STOCKS)
    if len(symbols) < NUM_STOCKS:
        logger.warning("Only got %d stocks, proceeding with available", len(symbols))

    # -------------------------------------------------------------------------
    # 2. 数据加载（带本地缓存）
    # -------------------------------------------------------------------------
    logger.info("[2/7] 从 AKShare 加载数据（首次约 %d 分钟）...", len(symbols) // 60 + 1)
    source = CachedAKShareDataSource(cache_dir=CACHE_DIR, adjust="qfq", provider="sina")
    loader = DataLoader(source)

    # DataHandler 直接负责加载和清洗
    handler = DataHandler(
        data_loader=loader,
        symbols=symbols,
        start_date=START_DATE,
        end_date=END_DATE,
        features=["ma_20", "rsi_14", "volatility_20d"],
        label="return_5d",
        cleaner=DataCleaner(),
        pre_processor=compute_factors,
        na_method="drop",
    )

    df = handler.get_data()
    logger.info("合并后共 %d 条样本 (%.1f 万), %d 只股票",
                len(df), len(df) / 10000, df.index.get_level_values("symbol").nunique())

    # -------------------------------------------------------------------------
    # 3. 时序拆分训练 / 测试
    # -------------------------------------------------------------------------
    logger.info("[3/7] 时序拆分训练集 / 测试集...")
    train_df = df.loc[df.index.get_level_values(0) <= TRAIN_END]
    test_df = df.loc[df.index.get_level_values(0) > TRAIN_END]
    logger.info("训练集: %d 条, 测试集: %d 条", len(train_df), len(test_df))

    feature_cols = handler.get_feature_cols()
    label_col = handler.get_label_col()

    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    # -------------------------------------------------------------------------
    # 4. 模型训练
    # -------------------------------------------------------------------------
    logger.info("[4/7] 训练 LightGBM 模型...")
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
        "num_threads": 4,
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
    # 5. 回测（截面多股票）
    # -------------------------------------------------------------------------
    logger.info("[5/7] 回测模拟（截面多股票）...")
    test_df = test_df.copy()
    test_df["pred"] = preds

    # 将长格式 pivot 为截面矩阵: index=date, columns=symbol
    pred_pivot = test_df["pred"].unstack(level="symbol")
    price_pivot = test_df["close"].unstack(level="symbol")

    # 只保留同时有预测和价格的日期
    common_dates = pred_pivot.index.intersection(price_pivot.index)
    pred_pivot = pred_pivot.loc[common_dates]
    price_pivot = price_pivot.loc[common_dates]

    # 填充缺失值（停牌股票用 NaN 表示不可交易）
    pred_pivot = pred_pivot.ffill(limit=5)
    price_pivot = price_pivot.ffill()

    logger.info("回测日期范围: %s ~ %s (%d 个交易日)",
                common_dates[0].strftime("%Y-%m-%d"),
                common_dates[-1].strftime("%Y-%m-%d"),
                len(common_dates))

    engine = BacktestEngine(
        strategy=TopKStrategy(k=10),  # 每日选预测最高的10只
        executor=Executor(commission_rate=0.001, min_commission=0),
        initial_capital=10_000_000,
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
    # 6. 记录实验结果
    # -------------------------------------------------------------------------
    logger.info("[6/7] 保存实验记录...")
    recorder = Recorder(EXPERIMENT_DIR, experiment_name="large_scale_demo")
    recorder.log_params({
        "num_stocks": len(symbols),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "train_end": TRAIN_END,
        "features": feature_cols,
        "label": label_col,
        "model": "LightGBM",
        "top_k": 10,
    })
    recorder.log_metrics({**val_metrics, **metrics})
    recorder.log_artifact("predictions.csv", pd.Series(preds, index=y_test.index, name="prediction"))
    recorder.log_artifact("portfolio_values.csv", result["portfolio_values"])
    logger.info("结果已保存到: %s", recorder.get_run_dir())

    # -------------------------------------------------------------------------
    # 7. 收益摘要
    # -------------------------------------------------------------------------
    logger.info("[7/7] 收益摘要:")
    returns = result["returns"]
    logger.info("  日均收益:   %.4f%%", returns.mean() * 100)
    logger.info("  收益标准差: %.4f%%", returns.std() * 100)
    logger.info("  正收益天数: %d / %d (%.1f%%)",
                (returns > 0).sum(), len(returns), (returns > 0).mean() * 100)

    logger.info("=" * 70)
    logger.info("Demo 完成!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
