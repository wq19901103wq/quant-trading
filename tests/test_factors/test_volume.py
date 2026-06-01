import pandas as pd
import pytest
from quant_trading.factors.volume import VolumeMAFactor, OBVFactor


class TestVolumeMAFactor:
    def test_volume_ma_shifted(self):
        df = pd.DataFrame({"volume": [100, 200, 300, 400, 500]})
        f = VolumeMAFactor(window=3)
        result = f.compute(df)
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == pytest.approx(200.0)
        assert result.iloc[4] == pytest.approx(300.0)
