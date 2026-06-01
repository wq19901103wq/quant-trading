"""因子注册中心."""
from typing import Dict, Type, List
from quant_trading.factors.base import Factor


class FactorRegistry:
    _registry: Dict[str, Type[Factor]] = {}

    @classmethod
    def register(cls, factor_cls: Type[Factor]) -> Type[Factor]:
        cls._registry[factor_cls.name] = factor_cls
        return factor_cls

    @classmethod
    def get(cls, name: str) -> Type[Factor]:
        if name not in cls._registry:
            raise KeyError(f"Factor '{name}' not registered")
        return cls._registry[name]

    @classmethod
    def list_factors(cls) -> List[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def clear(cls):
        cls._registry.clear()
