"""
Model utilities: train, evaluate, save, load.

This version uses callbacks for early stopping so it's compatible
with LightGBM versions that don't accept early_stopping_rounds as a kwarg.
"""
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import warnings
from typing import Optional, Dict

def train_lightgbm(X_train, y_train, X_valid=None, y_valid=None, params: Optional[Dict]=None,
                   num_boost_round: int = 1000, early_stopping_rounds: int = 50, verbose_eval: int = 50):
    """
    Train a LightGBM model using callbacks for early stopping and logging.

    Args:
        X_train, y_train: training arrays / DataFrame / Series
        X_valid, y_valid: optional validation arrays
        params: lightgbm params dict
        num_boost_round: maximum boosting rounds
        early_stopping_rounds: early stopping patience
        verbose_eval: logging frequency (rounds)

    Returns:
        trained lightgbm Booster
    """
    if params is None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "seed": 42
        }

    dtrain = lgb.Dataset(X_train, label=y_train)

    valid_sets = [dtrain]
    valid_names = ["train"]
    if X_valid is not None and y_valid is not None:
        dvalid = lgb.Dataset(X_valid, label=y_valid)
        valid_sets.append(dvalid)
        valid_names.append("valid")

    # Use callbacks for early stopping and evaluation logging (works across LightGBM versions)
    callbacks = []
    try:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
    except Exception:
        # older/newer versions may require different usage; try fallback by warning and proceeding without callback
        warnings.warn("early_stopping callback not available in this LightGBM build; training without early stopping.", UserWarning)
        callbacks = []

    # logging callback (if supported)
    try:
        callbacks.append(lgb.log_evaluation(period=verbose_eval))
    except Exception:
        # ignore if not available
        pass

    # Call train with callbacks. Avoid passing early_stopping_rounds kwarg directly.
    model = lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks)
    return model

def evaluate(model, X, y, threshold=0.5):
    """
    Evaluate a lightgbm Booster or sklearn wrapper (predict_proba) on X,y.
    Returns a dict containing metrics and the raw probabilities.
    """
    # Get predicted probabilities
    try:
        # For Booster
        proba = model.predict(X)
    except Exception:
        # If model is LGBMClassifier sklearn wrapper
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            raise

    auc = float(roc_auc_score(y, proba))
    pr_auc = float(average_precision_score(y, proba))
    preds = (proba >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    cm = confusion_matrix(y, preds)
    return {"roc_auc": auc, "pr_auc": pr_auc, "precision": float(p), "recall": float(r), "f1": float(f), "confusion_matrix": cm, "proba": proba}

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)