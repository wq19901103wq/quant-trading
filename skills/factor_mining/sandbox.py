"""因子代码沙箱：安全执行 AI 生成的因子计算代码."""
import ast
import re
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class SandboxError(Exception):
    pass


class FactorSandbox:
    """在受限命名空间中安全执行因子代码.

    安全策略:
    1. AST 静态分析：禁止导入/调用危险模块和函数
    2. 受限 globals：只允许 pandas, numpy 和内置安全函数
    3. 超时控制：通过外部调用方控制（建议用 signal 或 threading）
    """

    _FORBIDDEN_IMPORTS = {
        "os", "sys", "subprocess", "shutil", "socket", "urllib", "http",
        "ftplib", "pickle", "marshal", "eval", "exec", "compile",
        "__import__", "open", "input", "raw_input", "breakpoint",
    }

    _FORBIDDEN_CALLS = {
        "eval", "exec", "compile", "__import__", "open",
        "breakpoint", "input", "raw_input",
    }

    _ALLOWED_MODULES = {"pandas", "numpy", "pd", "np"}

    def __init__(self):
        self.last_error = ""

    def validate_code(self, code: str) -> bool:
        """AST 静态分析，检查代码是否安全."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            self.last_error = f"Syntax error: {e}"
            return False

        for node in ast.walk(tree):
            # 禁止 import
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in self._FORBIDDEN_IMPORTS:
                        self.last_error = f"Forbidden import: {alias.name}"
                        return False

            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in self._FORBIDDEN_IMPORTS:
                    self.last_error = f"Forbidden import from: {node.module}"
                    return False

            # 禁止危险函数调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self._FORBIDDEN_CALLS:
                    self.last_error = f"Forbidden call: {node.func.id}"
                    return False

            # 禁止属性访问危险函数（如 os.system）
            if isinstance(node, ast.Attribute):
                if node.attr in self._FORBIDDEN_CALLS:
                    self.last_error = f"Forbidden attribute access: {node.attr}"
                    return False

        return True

    def execute(self, code: str, df: pd.DataFrame) -> Optional[pd.Series]:
        """在沙箱中执行因子代码，返回计算结果.

        Args:
            code: AI 生成的 Python 代码字符串
            df: 输入数据（MultiIndex DataFrame）

        Returns:
            因子计算结果（pd.Series），失败返回 None
        """
        if not self.validate_code(code):
            return None

        # 受限命名空间
        restricted_globals = {
            "pd": pd,
            "np": np,
            "df": df,
        }
        restricted_locals = {}

        try:
            exec(code, restricted_globals, restricted_locals)
        except Exception as e:
            self.last_error = f"Execution error: {type(e).__name__}: {e}"
            return None

        # 尝试从 locals 中提取结果
        # 期望代码中定义了 result 变量
        if "result" in restricted_locals:
            result = restricted_locals["result"]
        else:
            self.last_error = "No result produced"
            return None

        if not isinstance(result, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
            self.last_error = f"Result type {type(result)} not supported"
            return None

        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]

        if isinstance(result, np.ndarray):
            result = pd.Series(result, index=df.index)

        if isinstance(result, (list, tuple)):
            result = pd.Series(result, index=df.index)

        return result

    def get_error(self) -> str:
        return self.last_error
