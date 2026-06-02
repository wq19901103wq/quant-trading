import numpy as np
import pandas as pd
import pytest

from skills.factor_mining.sandbox import FactorSandbox
from skills.factor_mining.skill import FactorMiningSkill, DummyLLMClient
from skills.factor_mining.prompts import FactorPrompts
from skills.factor_mining.runner import FactorMiningRunner
from src.factors.registry import FactorRegistry


class TestFactorSandbox:
    def test_validate_safe_code(self):
        sandbox = FactorSandbox()
        code = "result = df['close'] * 2"
        assert sandbox.validate_code(code) is True

    def test_validate_forbidden_import(self):
        sandbox = FactorSandbox()
        code = "import os\nresult = df['close']"
        assert sandbox.validate_code(code) is False
        assert "Forbidden import" in sandbox.get_error()

    def test_validate_forbidden_call(self):
        sandbox = FactorSandbox()
        code = "result = eval('1+1')"
        assert sandbox.validate_code(code) is False
        assert "Forbidden call" in sandbox.get_error()

    def test_execute_simple_factor(self):
        sandbox = FactorSandbox()
        df = pd.DataFrame({
            "close": [10.0, 11.0, 12.0, 13.0, 14.0],
            "symbol": ["A"] * 5,
        }, index=pd.date_range("2020-01-01", periods=5))
        df = df.set_index([df.index, "symbol"])
        df.index.names = ["date", "symbol"]

        code = "result = df['close'] * 2"
        result = sandbox.execute(code, df)
        assert result is not None
        assert result.iloc[0] == 20.0

    def test_execute_no_result(self):
        sandbox = FactorSandbox()
        df = pd.DataFrame({"close": [1, 2, 3]})
        code = "x = 1 + 1"
        result = sandbox.execute(code, df)
        assert result is None
        assert "No result" in sandbox.get_error()


class TestFactorPrompts:
    def test_generate_factor_prompt(self):
        prompt = FactorPrompts.generate_factor(
            existing_factors=["ma_20", "rsi_14"],
            history=[{"name": "test", "ic": 0.01, "ir": 0.1}],
        )
        assert "ma_20" in prompt
        assert "rsi_14" in prompt
        assert "IC=0.0100" in prompt
        assert "result" in prompt

    def test_generate_factor_prompt_empty(self):
        prompt = FactorPrompts.generate_factor(existing_factors=[], history=[])
        assert "暂无" in prompt

    def test_system_prompt(self):
        prompt = FactorPrompts.system_prompt()
        assert "量化研究员" in prompt


class TestFactorMiningSkill:
    def test_extract_code_from_markdown(self):
        response = "```python\nresult = df['close'] * 2\n```"
        code = FactorMiningSkill._extract_code(response)
        assert code == "result = df['close'] * 2"

    def test_extract_code_without_markdown(self):
        response = "result = df['close'] * 2"
        code = FactorMiningSkill._extract_code(response)
        assert code == "result = df['close'] * 2"

    def test_dummy_llm_generates_code(self):
        client = DummyLLMClient(seed=42)
        response = client.chat("system", "user")
        assert "result" in response
        assert "```python" in response

    def test_skill_generate_code(self):
        skill = FactorMiningSkill(llm_client=DummyLLMClient(seed=42))
        FactorRegistry.clear()
        code = skill.generate_code(existing_factors=[])
        assert code is not None
        assert "result" in code

    def test_skill_evaluate_in_sandbox(self):
        skill = FactorMiningSkill(llm_client=DummyLLMClient(seed=42))
        df = pd.DataFrame({
            "close": [10.0, 11.0, 12.0, 13.0, 14.0] * 2,
            "symbol": ["A"] * 5 + ["B"] * 5,
        }, index=pd.date_range("2020-01-01", periods=10))
        df = df.set_index([df.index, "symbol"])
        df.index.names = ["date", "symbol"]

        forward_return = pd.Series(np.random.randn(10), index=df.index)
        code = "result = df.groupby(level='symbol')['close'].transform(lambda x: x.pct_change().shift(1))"

        result = skill.evaluate_in_sandbox(code, df, forward_return)
        assert result is not None
        assert "ic" in result
        assert "rank_ic" in result

    def test_skill_add_history(self):
        skill = FactorMiningSkill()
        skill.add_history({"name": "test", "ic": 0.05})
        assert len(skill.get_history()) == 1


class TestFactorMiningRunner:
    def test_runner_with_mock_data(self):
        from src.data.handler import DataHandler
        from src.data.loader import CSVDataSource, DataLoader
        import tempfile
        import os

        FactorRegistry.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试 CSV
            df = pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=20),
                "open": list(range(20, 40)),
                "high": [x + 2 for x in range(20, 40)],
                "low": [x - 2 for x in range(20, 40)],
                "close": list(range(20, 40)),
                "volume": [1000] * 20,
            })
            df.to_csv(os.path.join(tmpdir, "000001.csv"), index=False)

            loader = DataLoader(CSVDataSource(tmpdir))
            handler = DataHandler(
                data_loader=loader,
                symbols=["000001"],
                start_date="2020-01-01",
                end_date="2020-01-31",
                features=["close", "volume"],
                label="close",
            )

            runner = FactorMiningRunner(
                data_handler=handler,
                skill=FactorMiningSkill(llm_client=DummyLLMClient(seed=42)),
                max_iterations=2,
                max_consecutive_failures=5,
                ic_threshold=0.5,  # 设置很高，确保都 rejected（测试稳定性）
            )

            summary = runner.run()
            assert "total_iterations" in summary
            assert summary["total_iterations"] <= 2
