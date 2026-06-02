#!/usr/bin/env python3
"""Factor Mining Skill 示例：用 Kimi 大模型自动挖掘因子.

用法:
    cd ~/quant-trading
    conda run -n quant-trading python examples/run_factor_mining.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from skills.factor_mining.llm_clients import DeepSeekLLMClient
from skills.factor_mining.skill import FactorMiningSkill
from skills.factor_mining.runner import FactorMiningRunner
from src.data.loader import CachedAKShareDataSource, DataLoader
from src.data.cleaner import DataCleaner
from src.data.handler import DataHandler
from src.experiments.recorder import Recorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("factor_mining")


def compute_factors(df):
    """预计算基线因子."""
    import pandas as pd
    df = df.copy()
    df["ma_20"] = df.groupby(level="symbol")["close"].transform(
        lambda x: x.rolling(20).mean().shift(1)
    )
    df["rsi_14"] = df.groupby(level="symbol")["close"].transform(
        lambda x: 100 - (100 / (1 + x.diff().clip(lower=0).rolling(14).mean() / (-x.diff().clip(upper=0)).rolling(14).mean()))
    ).shift(1)
    df["volatility_20d"] = df.groupby(level="symbol")["close"].transform(
        lambda x: x.pct_change().rolling(20).std().shift(1)
    )
    df["return_5d"] = df.groupby(level="symbol")["close"].transform(
        lambda x: x.pct_change(5).shift(-5)
    )
    return df


def main():
    logger.info("=" * 60)
    logger.info("Factor Mining Skill Demo — Kimi LLM")
    logger.info("=" * 60)

    # 1. 加载数据（用已有缓存）
    logger.info("Loading data...")
    source = CachedAKShareDataSource(cache_dir="./data/cache", adjust="qfq", provider="sina")
    loader = DataLoader(source)

    handler = DataHandler(
        data_loader=loader,
        symbols=["000001", "000002", "600036"],  # 先小样本测试
        start_date="2022-01-01",
        end_date="2024-12-31",
        features=["ma_20", "rsi_14", "volatility_20d"],
        label="return_5d",
        cleaner=DataCleaner(),
        pre_processor=compute_factors,
        na_method="drop",
    )

    # 2. 初始化 Kimi client
    logger.info("Initializing Kimi LLM client...")
    try:
        llm = DeepSeekLLMClient(model="deepseek-chat")
    except ValueError as e:
        logger.error("Failed to init LLM client: %s", e)
        return

    skill = FactorMiningSkill(llm_client=llm)
    recorder = Recorder("./experiments", experiment_name="factor_mining_kimi")

    # 3. 运行因子挖掘
    logger.info("Running factor mining (max 3 iterations)...")
    runner = FactorMiningRunner(
        data_handler=handler,
        skill=skill,
        recorder=recorder,
        max_iterations=3,
        max_consecutive_failures=3,
        ic_threshold=0.01,
        ir_threshold=0.1,
    )

    summary = runner.run()
    logger.info("Summary: %s", summary)
    logger.info("Results saved to: %s", recorder.get_run_dir())


if __name__ == "__main__":
    main()
