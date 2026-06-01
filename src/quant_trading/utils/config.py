"""配置管理模块."""
import yaml
import os
from typing import Any


class Config:
    def __init__(self, config_dict: dict):
        self._config = config_dict

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def to_dict(self) -> dict:
        return self._config.copy()


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)


def save_config_snapshot(config: Config, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    config_path = os.path.join(log_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
