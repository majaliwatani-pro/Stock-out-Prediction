import pandas as pd
import numpy as np
from src.feature_engineering import build_features

def test_build_features_runs():
    # small sample
    data = {
        "date": pd.date_range("2023-01-01", periods=10, freq="D"),
        "store_id": [1]*10,
        "item_id": [1]*10,
        "sales": np.arange(10),
        "stock_on_hand": np.linspace(10,0,10)
    }
    df = pd.DataFrame(data)
    out = build_features(df)
    # check key features exist
    assert "sales_lag_1" in out.columns or "sales_lag_1" in out.columns
    assert "days_of_cover" in out.columns
