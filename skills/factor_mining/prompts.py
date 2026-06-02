"""Prompt 模板：因子生成和评估."""
from typing import List, Dict


class FactorPrompts:
    """封装大模型 prompt 模板."""

    @staticmethod
    def generate_factor(
        existing_factors: List[str],
        history: List[Dict],
        max_length: int = 800,
    ) -> str:
        """生成新因子的 prompt.

        Args:
            existing_factors: 已有因子名称列表
            history: 历史实验记录，每个元素是 dict(name, ic, ir, sharpe)
            max_length: 控制输出长度
        """
        history_str = ""
        if history:
            lines = []
            for record in history[-5:]:  # 只取最近 5 条
                ic = record.get('ic', 'N/A')
                ir = record.get('ir', 'N/A')
                ic_str = f"{ic:.4f}" if isinstance(ic, (int, float)) else str(ic)
                ir_str = f"{ir:.4f}" if isinstance(ir, (int, float)) else str(ir)
                lines.append(
                    f"- {record.get('name', 'unknown')}: "
                    f"IC={ic_str}, "
                    f"IR={ir_str}"
                )
            history_str = "\n".join(lines)
        else:
            history_str = "（暂无历史记录）"

        prompt = f"""你是一位量化研究员，擅长挖掘 A 股市场的有效 Alpha 因子。

## 任务
请生成一个全新的技术分析因子，用于预测股票未来 5 日收益。

## 已有因子（请勿重复）
{', '.join(existing_factors) if existing_factors else '（暂无）'}

## 最近实验记录
{history_str}

## 要求
1. 因子必须基于 `df` 计算，df 是一个 MultiIndex DataFrame，index 为 (date, symbol)
2. 可用列：open, high, low, close, volume
3. 必须包含 `shift(1)` 防止 lookahead（用未来数据）
4. 代码必须简洁，不超过 {max_length} 字符
5. 最终计算结果赋值给变量 `result`
6. 因子名称用英文，如 momentum_5d, volume_price_ratio 等

## 输出格式
只输出 Python 代码，不要任何解释文字。格式如下：

```python
result = df.groupby(level='symbol')['close'].transform(lambda x: x.pct_change(5).shift(1))
```

请生成一个与已有因子不同的新因子：
"""
        return prompt

    @staticmethod
    def evaluate_and_suggest(
        factor_name: str,
        ic: float,
        ir: float,
        sharpe: float,
        history: List[Dict],
    ) -> str:
        """根据因子表现生成改进建议的 prompt."""
        prompt = f"""因子 "{factor_name}" 的评估结果如下：
- IC (信息系数): {ic:.4f}
- IR (信息比率): {ir:.4f}
- Sharpe: {sharpe:.4f}

IC > 0.02 且 IR > 0.3 视为有效因子。

请分析该因子表现不佳的原因，并给出具体的改进方向（1-2句话即可）。
"""
        return prompt

    @staticmethod
    def system_prompt() -> str:
        """系统级 prompt，定义大模型角色."""
        return (
            "你是一位专业的量化研究员，精通 A 股市场技术分析和因子挖掘。"
            "你的任务是根据给定的股票数据生成有效的 Alpha 因子。"
            "你只输出 Python 代码，不输出任何解释文字。"
        )
