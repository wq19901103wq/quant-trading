"""因子基类."""
from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class Factor(ABC):
    name: str = ""
    dependencies: List[str] = []

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        pass

    def get_name(self) -> str:
        return self.name
