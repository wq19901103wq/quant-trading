"""Factor Mining Skill 主类：大模型驱动因子生成与评估."""
import os
import re
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np

from skills.factor_mining.sandbox import FactorSandbox
from skills.factor_mining.prompts import FactorPrompts
from src.factors.base import Factor
from src.factors.registry import FactorRegistry
from src.factors.evaluator import FactorEvaluator


class LLMClient:
    """大模型客户端抽象接口.

    使用者可以继承此类实现自己的客户端（OpenAI、Kimi、DeepSeek 等）。
    """

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """调用大模型，返回文本回复.

        Args:
            system_prompt: 系统级 prompt（定义角色）
            user_prompt: 用户级 prompt（具体任务）

        Returns:
            大模型生成的文本
        """
        raise NotImplementedError("Subclass must implement chat()")


class DummyLLMClient(LLMClient):
    """默认 dummy 客户端：不调用真实 API，返回随机因子代码（用于测试）."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._templates = [
            "result = df.groupby(level='symbol')['close'].transform(lambda x: x.pct_change(10).shift(1))",
            "result = df.groupby(level='symbol')['volume'].transform(lambda x: x.rolling(20).mean().shift(1) / x.shift(1))",
            "result = df.groupby(level='symbol')['close'].transform(lambda x: (x - x.rolling(10).min()) / (x.rolling(10).max() - x.rolling(10).min() + 1e-8)).shift(1)",
            "result = df.groupby(level='symbol')['close'].transform(lambda x: x.diff(5).shift(1))",
            "result = df.groupby(level='symbol')['volume'].transform(lambda x: x.pct_change().shift(1)) * df.groupby(level='symbol')['close'].transform(lambda x: x.pct_change().shift(1))",
        ]

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        template = self.rng.choice(self._templates)
        return f"```python\n{template}\n```"


class FactorMiningSkill:
    """因子挖掘 Skill：协调大模型生成、沙箱执行、因子评估."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        sandbox: Optional[FactorSandbox] = None,
        evaluator: Optional[FactorEvaluator] = None,
    ):
        self.llm = llm_client or DummyLLMClient()
        self.sandbox = sandbox or FactorSandbox()
        self.evaluator = evaluator or FactorEvaluator()
        self.prompts = FactorPrompts()
        self.history: List[Dict] = []

    def generate_code(self, existing_factors: List[str]) -> Optional[str]:
        """调用大模型生成因子代码.

        Returns:
            提取出的 Python 代码字符串，失败返回 None
        """
        prompt = self.prompts.generate_factor(
            existing_factors=existing_factors,
            history=self.history,
        )
        try:
            response = self.llm.chat(
                system_prompt=self.prompts.system_prompt(),
                user_prompt=prompt,
            )
        except Exception as e:
            print(f"[LLM Error] {e}")
            return None

        # 从 markdown 代码块中提取 Python 代码
        code = self._extract_code(response)
        return code

    def evaluate_in_sandbox(
        self,
        code: str,
        df: pd.DataFrame,
        forward_return: pd.Series,
    ) -> Optional[Dict]:
        """在沙箱中执行因子代码并评估.

        Args:
            code: Python 代码字符串
            df: 输入数据
            forward_return: 未来收益序列（用于计算 IC）

        Returns:
            评估结果 dict，失败返回 None
        """
        factor_values = self.sandbox.execute(code, df)
        if factor_values is None:
            print(f"[Sandbox Error] {self.sandbox.get_error()}")
            return None

        # 规范化 index：确保 factor_values 和 forward_return 都是 (date, symbol) MultiIndex
        if not isinstance(factor_values.index, pd.MultiIndex):
            print(f"[Sandbox Error] Factor result index is not MultiIndex: {type(factor_values.index)}")
            return None

        # 对齐并计算 IC
        try:
            metrics = self.evaluator.evaluate(factor_values, forward_return)
        except Exception as e:
            print(f"[Evaluator Error] {e}")
            return None

        return {
            "code": code,
            "ic": metrics.get("ic", np.nan),
            "rank_ic": metrics.get("rank_ic", np.nan),
            "ic_ir": metrics.get("ic_ir", np.nan),
            "factor_values": factor_values,
        }

    def register_factor(self, name: str, code: str) -> bool:
        """将因子注册到 FactorRegistry（动态创建 Factor 子类）.

        Args:
            name: 因子名称
            code: 因子计算代码

        Returns:
            是否注册成功
        """
        # 动态创建 Factor 子类
        def compute(self, df: pd.DataFrame) -> pd.Series:
            return FactorSandbox().execute(code, df)

        factor_cls = type(
            name,
            (Factor,),
            {"name": name, "dependencies": ["close"], "compute": compute},
        )

        try:
            FactorRegistry.register(factor_cls)
            return True
        except Exception as e:
            print(f"[Registry Error] {e}")
            return False

    def add_history(self, record: Dict):
        """添加实验记录."""
        self.history.append(record)

    def get_history(self) -> List[Dict]:
        """获取全部实验历史."""
        return self.history.copy()

    @staticmethod
    def _extract_code(response: str) -> Optional[str]:
        """从 markdown 代码块中提取 Python 代码."""
        # 匹配 ```python ... ``` 或 ``` ... ```
        patterns = [
            r"```python\n(.*?)\n```",
            r"```\n(.*?)\n```",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # 如果没有代码块，尝试直接返回（假设整段都是代码）
        cleaned = response.strip()
        if "=" in cleaned and "result" in cleaned:
            return cleaned

        return None
