import pytest
from src.utils.config import Config, load_config, save_config_snapshot
import tempfile
import os


class TestConfig:
    def test_config_dot_access(self):
        cfg = Config({"model": {"max_depth": 5, "lr": 0.05}})
        assert cfg.get("model.max_depth") == 5
        assert cfg.get("model.lr") == 0.05

    def test_config_default_value(self):
        cfg = Config({})
        assert cfg.get("missing.key", "default") == "default"

    def test_load_and_save_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                f.write("model:\n  max_depth: 5\n")
            
            cfg = load_config(config_path)
            assert cfg.get("model.max_depth") == 5
            
            log_dir = os.path.join(tmpdir, "logs")
            save_config_snapshot(cfg, log_dir)
            assert os.path.exists(os.path.join(log_dir, "config.yaml"))
