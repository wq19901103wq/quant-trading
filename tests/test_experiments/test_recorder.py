import os
import tempfile
import json
import pandas as pd
from src.experiments.recorder import Recorder


class TestRecorder:
    def test_log_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = Recorder(tmpdir, "test_exp")
            rec.log_params({"lr": 0.01, "epochs": 100})
            path = os.path.join(rec.get_run_dir(), "params.json")
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["lr"] == 0.01

    def test_log_artifact_dataframe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = Recorder(tmpdir, "test_exp")
            df = pd.DataFrame({"a": [1, 2, 3]})
            rec.log_artifact("data.csv", df)
            path = os.path.join(rec.get_run_dir(), "data.csv")
            assert os.path.exists(path)
            loaded = pd.read_csv(path, index_col=0)
            assert loaded["a"].tolist() == [1, 2, 3]
