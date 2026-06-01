import pandas as pd
import numpy as np
import pytest
from src.models.base import Model
import tempfile
import os


class DummyModel(Model):
    def __init__(self):
        self.coef = 1.0

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X.iloc[:, 0].values * self.coef

    def get_params(self):
        return {"coef": self.coef}


class TestModelBase:
    def test_save_and_load(self):
        model = DummyModel()
        model.coef = 2.5
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            model.save(path)
            loaded = Model.load(path)
            assert isinstance(loaded, DummyModel)
            assert loaded.coef == 2.5
