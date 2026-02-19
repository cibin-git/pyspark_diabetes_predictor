"""Simple FastAPI model server that loads `models/model.pkl`."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Model Serve")


class PredictRequest(BaseModel):
    features: list


@app.on_event("startup")
def load_model():
    try:
        app.state.model = joblib.load("models/model.pkl")
    except Exception:
        app.state.model = None


@app.get("/health")
def health():
    return {"ready": app.state.model is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded; run training first")
    arr = np.array(req.features)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    preds = app.state.model.predict(arr)
    return {"predictions": preds.tolist()}
