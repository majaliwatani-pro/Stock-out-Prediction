"""
Feature engineering for stock-out prediction (fixed).
This version avoids calling reset_index(level=...) on Series without a MultiIndex.
Uses groupby.transform for group-wise rolling features so results are aligned with the original index.
"""
from typing import Iterable, List
import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    # weekofyear: use isocalendar().week for pandas >= 1.1
    df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)
    return df

def create_lag_roll_features(
    df: pd.DataFrame,
    group_cols: List[str] = ["store_id", "item_id"],
    target_col: str = "sales",
    lags: Iterable[int] = (1, 7, 14),
    windows: Iterable[int] = (7, 14, 28),
) -> pd.DataFrame:
    """
    Create lag and rolling features.
    - lags: simple groupby.shift(lag)
    - rolling means/std: groupby.transform(lambda s: s.shift(1).rolling(window=w).mean())
      shift(1) prevents leakage (we don't include current day's sales when computing historical window)
    """
    df = df.copy()
    # Ensure rows are sorted by group + date for shift/rolling correctness
    sort_cols = list(group_cols) + ["date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # LAGS
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)

    # ROLLING MEANS / STDS (use transform so result aligns with original index)
    for w in windows:
        df[f"{target_col}_rmean_{w}"] = df.groupby(group_cols)[target_col].transform(
            lambda s, w=w: s.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f"{target_col}_rstd_{w}"] = df.groupby(group_cols)[target_col].transform(
            lambda s, w=w: s.shift(1).rolling(window=w, min_periods=1).std().fillna(0)
        )

    return df

def compute_days_of_cover(df: pd.DataFrame, stock_col: str = "stock_on_hand", avg_col: str = "sales_rmean_7") -> pd.DataFrame:
    df = df.copy()
    if stock_col in df.columns and avg_col in df.columns:
        df["days_of_cover"] = df[stock_col] / (df[avg_col] + 1e-6)
        df["days_of_cover"] = df["days_of_cover"].replace([np.inf, -np.inf], np.nan).fillna(-1)
    else:
        df["days_of_cover"] = -1
    return df

def add_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"store_id", "item_id", "sales"}.issubset(df.columns):
        df["store_item_mean"] = df.groupby(["store_id", "item_id"])["sales"].transform("mean")
    else:
        df["store_item_mean"] = -1
    if {"store_id", "sales"}.issubset(df.columns):
        df["store_mean_item"] = df.groupby(["store_id"])["sales"].transform("mean")
    else:
        df["store_mean_item"] = -1
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    High-level pipeline to produce a feature dataframe.
    """
    df = df.copy()
    df = add_time_features(df)
    df = create_lag_roll_features(df)
    df = compute_days_of_cover(df)
    df = add_aggregations(df)
    # fill na/infs with sentinel
    df = df.replace([np.inf, -np.inf], np.nan).fillna(-1)
    return df