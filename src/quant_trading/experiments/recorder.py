"""实验记录器."""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd


class Recorder:
    def __init__(self, experiment_dir: str, experiment_name: Optional[str] = None):
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = os.path.join(experiment_dir, self.experiment_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.records: Dict[str, Any] = {}

    def log_params(self, params: Dict[str, Any]) -> None:
        self.records["params"] = params
        self._save_json("params.json", params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self.records["metrics"] = metrics
        self._save_json("metrics.json", metrics)

    def log_artifact(self, name: str, data: Any) -> None:
        path = os.path.join(self.run_dir, name)
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=True)
        elif isinstance(data, pd.Series):
            data.to_csv(path, index=True)
        else:
            with open(path, "w") as f:
                if isinstance(data, dict):
                    json.dump(data, f, indent=2, default=str)
                else:
                    f.write(str(data))

    def _save_json(self, filename: str, data: Dict) -> None:
        path = os.path.join(self.run_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_run_dir(self) -> str:
        return self.run_dir
