from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---- Paths ----
BUNDLE_PATH = Path("artifacts") / "model_bundle.joblib"

app = FastAPI(title="Store Sales Forecast API", version="0.1.0")


# ---- Request schema ----
class PredictRequest(BaseModel):
    onpromotion: float = Field(..., ge=0)
    is_holiday: int = Field(..., ge=0, le=1)
    lag_1: float = Field(..., ge=0)
    lag_7: float = Field(..., ge=0)
    rolling_7_mean: float = Field(..., ge=0)
    rolling_14_mean: float = Field(..., ge=0)


# ---- Load model once at startup ----
MODEL = None
FEATURES = None


@app.on_event("startup")
def load_model():
    global MODEL, FEATURES
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Model bundle not found: {BUNDLE_PATH.resolve()}")
    bundle = joblib.load(BUNDLE_PATH)
    MODEL = bundle["model"]
    FEATURES = bundle.get("features")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "bundle_exists": BUNDLE_PATH.exists(),
        "model_loaded": MODEL is not None,
        "features": FEATURES,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None:
        return {"error": "Model not loaded"}

    x = np.array(
        [[
            req.onpromotion,
            req.is_holiday,
            req.lag_1,
            req.lag_7,
            req.rolling_7_mean,
            req.rolling_14_mean,
        ]],
        dtype=float
    )

    y = float(MODEL.predict(x)[0])
    return {"prediction_sales": y}
