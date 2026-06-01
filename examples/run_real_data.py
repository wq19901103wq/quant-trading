#!/usr/bin/env python3
"""端到端 Demo：扩展窗口滚动回测 + 超额收益（500只股票 / 10年）.

滚动设计（扩展窗口）：
  Fold 1: 训练 2015-2017 → 测试 2018
  Fold 2: 训练 2015-2018 → 测试 2019
  Fold 3: 训练 2015-2019 → 测试 2020
  ...
  Fold 7: 训练 2015-2023 → 测试 2024

用法:
    cd ~/quant-trading
    conda run -n quant-trading python examples/run_real_data.py
"""
import logging
import time
from typing import Dict
import pandas as pd
import numpy as np

from src.data.loader import CachedAKShareDataSource, DataLoader
from src.data.cleaner import DataCleaner
from src.data.handler import DataHandler
from src.data.preprocessing import calculate_ma, calculate_rsi
from src.models.gbdt import LightGBMModel
from src.portfolio.strategy import TopKStrategy
from src.backtest.executor import Executor
from src.backtest.engine import BacktestEngine
from src.experiments.recorder import Recorder
from src.metrics.core import calculate_metrics, calculate_excess_metrics
from src.models.validation import evaluate_regression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo")

# =============================================================================
# 配置参数
# =============================================================================
NUM_STOCKS = 500
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
CACHE_DIR = "./data/cache"
EXPERIMENT_DIR = "./experiments"
INITIAL_CAPITAL = 10_000_000
TOP_K = 10


def get_stock_list(n: int = 500) -> list:
    """获取A股列表，过滤ST股，返回前n只."""
    import akshare as ak
    df = ak.stock_info_a_code_name()
    df = df[~df["name"].str.contains(r"ST|退|摘", na=False, regex=True)]
    df = df[df["code"].str.match(r"^[06\d]\d{5}$")]
    symbols = df["code"].head(n).tolist()
    logger.info("Selected %d stocks", len(symbols))
    return symbols


def get_benchmark_from_data(df: pd.DataFrame) -> pd.Series:
    """从已有股票数据构造等权组合收益作为基准（无需额外网络请求）."""
    # 每日所有股票的等权收益
    daily_returns = df["close"].groupby(level="symbol").pct_change()
    benchmark = daily_returns.groupby(level=0).mean()  # 按日期等权平均
    return benchmark.dropna()


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """为每只股票独立计算因子（避免跨股票污染）."""
    df = df.copy()
    df["ma_20"] = df.groupby(level="symbol")["close"].transform(
        lambda x: calculate_ma(x, window=20, shift=True)
    )
    df["rsi_14"] = df.groupby(level="symbol")["close"].transform(
        lambda x: calculate_rsi(x, window=14, shift=True)
    )
    df["volatility_20d"] = df.groupby(level="symbol")["close"].transform(
        lambda x: x.pct_change().rolling(20).std().shift(1)
    )
    df["return_5d"] = df.groupby(level="symbol")["close"].transform(
        lambda x: x.pct_change(5).shift(-5)
    )
    return df


def run_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, fold_idx: int, benchmark_returns: pd.Series) -> Dict:
    """运行单个 fold：训练 → 预测 → 回测."""
    feature_cols = ["ma_20", "rsi_14", "volatility_20d"]
    label_col = "return_5d"

    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    # 训练
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

    # 验证指标
    val_metrics = evaluate_regression(y_test, preds)

    # 回测准备
    test_df = test_df.copy()
    test_df["pred"] = preds
    pred_pivot = test_df["pred"].unstack(level="symbol")
    price_pivot = test_df["close"].unstack(level="symbol")

    common_dates = pred_pivot.index.intersection(price_pivot.index)
    pred_pivot = pred_pivot.loc[common_dates]
    price_pivot = price_pivot.loc[common_dates]
    pred_pivot = pred_pivot.ffill(limit=5)
    price_pivot = price_pivot.ffill()

    # 回测
    engine = BacktestEngine(
        strategy=TopKStrategy(k=TOP_K),
        executor=Executor(commission_rate=0.001, min_commission=0),
        initial_capital=INITIAL_CAPITAL,
    )
    result = engine.run(pred_pivot, price_pivot)
    metrics = result["metrics"]

    # 超额收益
    port_returns = result["returns"]
    bench_aligned = benchmark_returns.reindex(port_returns.index).fillna(0)
    excess_metrics = calculate_excess_metrics(port_returns, bench_aligned)

    logger.info(
        "  Fold %d 完成 | train=%d test=%d | "
        "total=%+.2f%% excess=%+.2f%% IR=%.3f maxdd=%.2f%%",
        fold_idx,
        len(train_df),
        len(test_df),
        metrics["total_return"] * 100,
        excess_metrics["annualized_excess_return"] * 100,
        excess_metrics["information_ratio"],
        metrics["max_drawdown"] * 100,
    )

    return {
        "fold": fold_idx,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "metrics": metrics,
        "excess_metrics": excess_metrics,
        "val_metrics": val_metrics,
        "portfolio_values": result["portfolio_values"],
        "returns": port_returns,
        "predictions": pd.Series(preds, index=y_test.index),
    }


def main():
    logger.info("=" * 70)
    logger.info("Quant Trading Pipeline Demo — Expanding Window + Excess Return")
    logger.info("Config: %d stocks × %s ~ %s", NUM_STOCKS, START_DATE, END_DATE)
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # 1. 获取股票列表
    # -------------------------------------------------------------------------
    logger.info("[1/6] 获取股票列表...")
    symbols = get_stock_list(NUM_STOCKS)

    # -------------------------------------------------------------------------
    # 2. 加载数据
    # -------------------------------------------------------------------------
    logger.info("[2/6] 加载股票数据...")
    source = CachedAKShareDataSource(cache_dir=CACHE_DIR, adjust="qfq", provider="sina")
    loader = DataLoader(source)

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
    n_symbols = df.index.get_level_values("symbol").nunique()
    logger.info("总样本: %d 条, 股票数: %d", len(df), n_symbols)

    # -------------------------------------------------------------------------
    # 3. 构造基准（等权组合）
    # -------------------------------------------------------------------------
    logger.info("[3/6] 构造等权基准...")
    benchmark_returns = get_benchmark_from_data(df)
    logger.info("基准数据: %d 个交易日", len(benchmark_returns))

    # -------------------------------------------------------------------------
    # 4. 扩展窗口滚动回测
    # -------------------------------------------------------------------------
    logger.info("[4/6] 扩展窗口滚动回测...")
    dates = df.index.get_level_values(0).unique().sort_values()
    years = sorted(dates.year.unique())

    # 找到所有完整年份的测试年（从2018开始到2024）
    test_years = [y for y in years if y >= 2018]
    logger.info("共 %d 个 fold, 测试年份: %s", len(test_years), test_years)

    all_fold_results = []
    all_portfolio_values = []
    all_returns = []

    for fold_idx, test_year in enumerate(test_years, 1):
        train_end_date = dates[dates.year < test_year][-1] if (dates.year < test_year).any() else dates[0]
        test_mask = dates.year == test_year
        test_dates = dates[test_mask]

        if len(test_dates) == 0:
            continue

        train_mask = df.index.get_level_values(0) <= train_end_date
        test_mask_df = df.index.get_level_values(0).isin(test_dates)
        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask_df]

        if len(train_df) < 1000 or len(test_df) < 100:
            logger.warning("Fold %d 数据不足，跳过", fold_idx)
            continue

        logger.info(
            "Fold %d/%d: 训练 %s ~ %s (%d条) → 测试 %d (%d条)",
            fold_idx,
            len(test_years),
            dates[0].strftime("%Y-%m-%d"),
            train_end_date.strftime("%Y-%m-%d"),
            len(train_df),
            test_year,
            len(test_df),
        )

        fold_result = run_fold(train_df, test_df, fold_idx, benchmark_returns)
        all_fold_results.append(fold_result)

        # 累加收益曲线（将各 fold 组合收益拼接）
        all_returns.append(fold_result["returns"])

    # -------------------------------------------------------------------------
    # 5. 汇总全周期结果
    # -------------------------------------------------------------------------
    logger.info("[5/6] 汇总全周期结果...")
    combined_returns = pd.concat(all_returns).sort_index()
    combined_metrics = calculate_metrics(combined_returns)
    bench_aligned = benchmark_returns.reindex(combined_returns.index).fillna(0)
    combined_excess = calculate_excess_metrics(combined_returns, bench_aligned)

    logger.info("=" * 50)
    logger.info("全周期汇总 (%d 个交易日)", len(combined_returns))
    logger.info("=" * 50)
    logger.info("绝对收益:")
    logger.info("  总收益:       %+.2f%%", combined_metrics["total_return"] * 100)
    logger.info("  年化收益:     %+.2f%%", combined_metrics["annualized_return"] * 100)
    logger.info("  Sharpe:       %.3f", combined_metrics["sharpe_ratio"])
    logger.info("  最大回撤:     %.2f%%", combined_metrics["max_drawdown"] * 100)
    logger.info("超额收益 (vs 沪深300):")
    logger.info("  超额收益:     %+.2f%%", combined_excess["annualized_excess_return"] * 100)
    logger.info("  信息比率 IR:  %.3f", combined_excess["information_ratio"])
    logger.info("  跟踪误差:     %.2f%%", combined_excess["tracking_error"] * 100)
    logger.info("  超额回撤:     %.2f%%", combined_excess["excess_max_drawdown"] * 100)
    logger.info("  Beta:         %.3f", combined_excess["beta"])

    # 各 fold 明细
    logger.info("-" * 50)
    logger.info("各 Fold 明细:")
    for r in all_fold_results:
        logger.info(
            "  Fold %d (%d): total=%+.2f%% excess=%+.2f%% IR=%.3f sharpe=%.3f",
            r["fold"],
            r["test_size"],
            r["metrics"]["total_return"] * 100,
            r["excess_metrics"]["annualized_excess_return"] * 100,
            r["excess_metrics"]["information_ratio"],
            r["metrics"]["sharpe_ratio"],
        )

    # -------------------------------------------------------------------------
    # 6. 保存实验记录
    # -------------------------------------------------------------------------
    logger.info("[6/6] 保存实验记录...")
    recorder = Recorder(EXPERIMENT_DIR, experiment_name="expanding_window_demo")
    recorder.log_params({
        "num_stocks": n_symbols,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "top_k": TOP_K,
        "initial_capital": INITIAL_CAPITAL,
        "strategy": "TopKStrategy",
        "model": "LightGBM",
        "window_type": "expanding",
        "test_years": test_years,
    })
    recorder.log_metrics({**combined_metrics, **combined_excess})
    recorder.log_artifact("returns.csv", combined_returns)
    recorder.log_artifact("fold_summary.json", {
        "folds": [
            {
                "fold": r["fold"],
                "train_size": r["train_size"],
                "test_size": r["test_size"],
                "total_return": r["metrics"]["total_return"],
                "annualized_return": r["metrics"]["annualized_return"],
                "sharpe": r["metrics"]["sharpe_ratio"],
                "max_drawdown": r["metrics"]["max_drawdown"],
                "excess_return": r["excess_metrics"]["annualized_excess_return"],
                "information_ratio": r["excess_metrics"]["information_ratio"],
                "beta": r["excess_metrics"]["beta"],
            }
            for r in all_fold_results
        ]
    })
    logger.info("结果已保存到: %s", recorder.get_run_dir())

    logger.info("=" * 70)
    logger.info("Demo 完成!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
