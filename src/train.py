"""
Training entrypoint.

Usage:
python src/train.py --data-path data/sample_data.csv --model-out models/stockout_model.pkl --horizon 7
"""
import argparse
import os
import pandas as pd
from data_prep import prepare_dataset, create_label
from feature_engineering import build_features
from model import train_lightgbm, evaluate, save_model
import numpy as np

def main(args):
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    df = pd.read_csv(args.data_path)
    df = prepare_dataset(df)
    df = create_label(df, horizon=args.horizon)
    df = build_features(df)
    # drop rows without label (initial rows per group)
    df = df.dropna(subset=["label"])
    # choose features
    exclude = {"date","label","store_id","item_id"}
    features = [c for c in df.columns if c not in exclude]
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(-1)
    y = df["label"].astype(int)
    # time-wise split (80/20)
    cutoff = df["date"].quantile(0.80)
    train_mask = df["date"] <= cutoff
    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[~train_mask], y[~train_mask]
    print("Training rows:", X_train.shape[0], "Validation rows:", X_valid.shape[0])
    model = train_lightgbm(X_train, y_train, X_valid, y_valid)
    print("Evaluating...")
    eval_train = evaluate(model, X_train, y_train)
    eval_valid = evaluate(model, X_valid, y_valid)
    print("Train metrics:", eval_train)
    print("Valid metrics:", eval_valid)
    save_model(model, args.model_out)
    print("Model saved:", args.model_out)
    # Save feature list for scoring
    feat_path = os.path.splitext(args.model_out)[0] + "_features.txt"
    with open(feat_path, "w") as f:
        f.write("\n".join(features))
    print("Saved feature list to:", feat_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-out", default="models/stockout_model.pkl")
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()
    main(args)
