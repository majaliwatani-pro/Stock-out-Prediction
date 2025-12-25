"""
FastAPI app for model inference.

Run from project root (recommended):
    python -m uvicorn src.predict_api:app --reload

This file uses package-relative imports so it works when started as a module.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from .model import load_model  # package-relative import
import json

MODEL_PATH = os.environ.get("MODEL_PATH", "models/stockout_model.pkl")
FEATURES_PATH = os.path.splitext(MODEL_PATH)[0] + "_features.txt"

app = FastAPI(title="Stock-out Prediction API")
model = None
FEATURES = None

class PredictPayload(BaseModel):
    store_id: int
    item_id: int
    date: str
    sales_lag_1: float = None
    sales_rmean_7: float = None
    days_of_cover: float = None
    on_promotion: int = 0
    price: float = None

@app.on_event("startup")
def startup_event():
    global model, FEATURES
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Train and place model there before starting API.")
    model = load_model(MODEL_PATH)
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            FEATURES = [l.strip() for l in f.readlines() if l.strip()]
    else:
        FEATURES = None

@app.post("/predict")
def predict(payload: PredictPayload):
    global model, FEATURES
    d = payload.dict()
    if FEATURES is None:
        raise HTTPException(status_code=500, detail="Feature list missing. Train the model and include features file.")
    # Build row matching saved features; fill missing with -1
    row = {}
    for feat in FEATURES:
        row[feat] = d.get(feat, -1)
    X = pd.DataFrame([row])
    try:
        proba = float(model.predict(X)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}")
    threshold = 0.5
    return {"stockout_probability": proba, "predicted_stockout": bool(proba >= threshold)}