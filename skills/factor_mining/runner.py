"""Factor Mining Runner：自动化迭代循环."""
import time
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from skills.factor_mining.skill import FactorMiningSkill, LLMClient
from skills.factor_mining.sandbox import FactorSandbox
from src.data.handler import DataHandler
from src.experiments.recorder import Recorder


class FactorMiningRunner:
    """自动化因子挖掘 runner.

    工作流:
    1. 从 DataHandler 获取数据
    2. 循环调用 FactorMiningSkill 生成因子
    3. 沙箱验证 + 评估
    4. 保留/丢弃判断
    5. 记录实验结果
    6. 终止条件检查
    """

    def __init__(
        self,
        data_handler: DataHandler,
        skill: Optional[FactorMiningSkill] = None,
        recorder: Optional[Recorder] = None,
        ic_threshold: float = 0.02,
        ir_threshold: float = 0.3,
        max_iterations: int = 10,
        max_consecutive_failures: int = 3,
    ):
        self.data_handler = data_handler
        self.skill = skill or FactorMiningSkill()
        self.recorder = recorder
        self.ic_threshold = ic_threshold
        self.ir_threshold = ir_threshold
        self.max_iterations = max_iterations
        self.max_consecutive_failures = max_consecutive_failures

        self.accepted_factors: List[Dict] = []
        self.rejected_factors: List[Dict] = []
        self.consecutive_failures = 0

    def run(self) -> Dict:
        """运行自动化因子挖掘循环.

        Returns:
            汇总结果 dict
        """
        print("=" * 60)
        print("Factor Mining Runner")
        print("=" * 60)

        df = self.data_handler.get_data()
        features = self.data_handler.get_feature_cols()
        label_col = self.data_handler.get_label_col()

        # 提取 forward_return 用于评估
        forward_return = df[label_col]

        # 已有因子列表（避免重复）
        from src.factors.registry import FactorRegistry
        existing_factors = FactorRegistry.list_factors()

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n[Iteration {iteration}/{self.max_iterations}]")

            # Step 1: 生成因子代码
            code = self.skill.generate_code(existing_factors)
            if code is None:
                print("  Failed to generate code")
                self.consecutive_failures += 1
                if self._should_stop():
                    break
                continue

            print(f"  Generated code: {code[:80]}...")

            # Step 2: 沙箱验证 + 评估
            result = self.skill.evaluate_in_sandbox(code, df, forward_return)
            if result is None:
                print(f"  Sandbox execution failed: {self.skill.sandbox.get_error()}")
                self.consecutive_failures += 1
                if self._should_stop():
                    break
                continue

            ic = result.get("ic", np.nan)
            rank_ic = result.get("rank_ic", np.nan)
            ic_ir = result.get("ic_ir", np.nan)

            print(f"  IC={ic:.4f}, RankIC={rank_ic:.4f}, IC_IR={ic_ir:.4f}")

            # Step 3: 保留/丢弃判断
            record = {
                "iteration": iteration,
                "code": code,
                "ic": ic,
                "rank_ic": rank_ic,
                "ic_ir": ic_ir,
            }

            if not np.isnan(ic) and ic > self.ic_threshold and not np.isnan(ic_ir) and ic_ir > self.ir_threshold:
                print(f"  ✅ ACCEPTED (IC>{self.ic_threshold}, IR>{self.ir_threshold})")
                self.accepted_factors.append(record)
                self.skill.add_history(record)
                self.consecutive_failures = 0

                # 可选：注册到 FactorRegistry
                factor_name = f"mined_factor_{iteration}"
                self.skill.register_factor(factor_name, code)
                existing_factors.append(factor_name)
            else:
                print(f"  ❌ REJECTED")
                self.rejected_factors.append(record)
                self.skill.add_history(record)
                self.consecutive_failures += 1

            # Step 4: 记录实验
            if self.recorder:
                self.recorder.log_artifact(f"factor_{iteration}.json", record)

            # Step 5: 检查终止条件
            if self._should_stop():
                print(f"\n  Stopping: consecutive failures = {self.consecutive_failures}")
                break

            # 节流，避免过快调用大模型 API
            time.sleep(0.5)

        # 汇总
        summary = {
            "total_iterations": iteration,
            "accepted": len(self.accepted_factors),
            "rejected": len(self.rejected_factors),
            "accepted_factors": self.accepted_factors,
            "rejected_factors": self.rejected_factors,
        }

        if self.recorder:
            self.recorder.log_metrics({
                "total_iterations": iteration,
                "accepted_count": len(self.accepted_factors),
                "rejected_count": len(self.rejected_factors),
            })

        print("\n" + "=" * 60)
        print(f"Done: {len(self.accepted_factors)} accepted, {len(self.rejected_factors)} rejected")
        print("=" * 60)

        return summary

    def _should_stop(self) -> bool:
        """检查是否满足终止条件."""
        return self.consecutive_failures >= self.max_consecutive_failures
