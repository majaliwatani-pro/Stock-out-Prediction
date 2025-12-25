"""
Data preparation & labeling utilities for stock-out prediction.

Provides:
- prepare_dataset(df): ensure types & sorting
- create_label(df, horizon=7, ...): create binary stock-out label

Usage:
    from data_prep import prepare_dataset, create_label
"""
import pandas as pd
import numpy as np
from typing import List

__all__ = ["prepare_dataset", "create_label"]

def prepare_dataset(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Ensure date is datetime, sort rows by group + date, reset index.
    Returns a new DataFrame (does not modify input in-place).
    """
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Expected column '{date_col}' in dataframe")
    df[date_col] = pd.to_datetime(df[date_col])
    # basic sanity: required group cols if present
    # Keep as-is; sorting will be done where group columns exist in downstream functions
    return df.sort_values(["date"]).reset_index(drop=True)

def create_label(
    df: pd.DataFrame,
    horizon: int = 7,
    group_cols: List[str] = ["store_id", "item_id"],
    stock_col: str = "stock_on_hand",
    sales_col: str = "sales",
) -> pd.DataFrame:
    """
    Create binary label 'label' indicating whether a stock-out occurs within the next `horizon` days.
    Logic:
      - compute stock at t+1 by shifting stock_col by -1
      - compute future_stock_min = rolling min of shifted stock over next `horizon` days (group-wise)
      - compute expected next-day demand = sales shifted by -1 (you can replace with a rolling mean)
      - label = 1 where future_stock_min <= expected_next_day_demand

    Returns DataFrame with a new 'label' column (int 0/1). Does not drop rows.
    """
    # Validate columns
    missing = [c for c in group_cols + [stock_col, sales_col, "date"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for create_label: {missing}")

    # Work on a copy and ensure proper sorting
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(group_cols + ["date"]).reset_index(drop=True)

    # 1) shifted next-day stock (aligned with current row)
    df["_shifted_stock_t_plus_1"] = df.groupby(group_cols)[stock_col].shift(-1)

    # 2) group-wise rolling min over the horizon on the shifted stock
    # transform returns aligned Series so no index gymnastics needed
    df["future_stock_min"] = df.groupby(group_cols)["_shifted_stock_t_plus_1"].transform(
        lambda s: s.rolling(window=horizon, min_periods=1).min()
    )

    # 3) expected next-day demand (simple next-day sales; replace with rolling mean if desired)
    df["expected_demand_next_day"] = df.groupby(group_cols)[sales_col].shift(-1)

    # 4) label: future stock that can't satisfy next-day demand => stock-out in horizon
    df["label"] = 0
    cond = df["future_stock_min"].notna() & (df["future_stock_min"] <= df["expected_demand_next_day"].fillna(0))
    df.loc[cond, "label"] = 1

    # cleanup temporary columns
    df.drop(columns=[c for c in ["_shifted_stock_t_plus_1", "future_stock_min", "expected_demand_next_day"] if c in df.columns], inplace=True)

    # ensure label is integer
    df["label"] = df["label"].astype(int)

    return df