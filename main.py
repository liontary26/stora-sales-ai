from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from pathlib import Path

# Model bundle (eğitimden sonra oluşmalı)
BUNDLE_PATH = Path("artifacts") / "model_bundle.joblib"

app = FastAPI(title="Store Sales Forecast API")

class PredictRequest(BaseModel):
    onpromotion: float = Field(..., ge=0)
    is_holiday: float = Field(..., ge=0, le=1)
    lag_1: float = Field(..., ge=0)
    lag_7: float = Field(..., ge=0)
    rolling_7_mean: float = Field(..., ge=0)
    rolling_14_mean: float = Field(..., ge=0)

@app.get("/health")
def health():
    return {"status": "ok", "bundle_exists": BUNDLE_PATH.exists()}

@app.post("/predict")
def predict(req: PredictRequest):
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle["model"]
    # bundle["features"] kullanmak istersen: features = bundle["features"]

    x = [[
        req.onpromotion,
        req.is_holiday,
        req.lag_1,
        req.lag_7,
        req.rolling_7_mean,
        req.rolling_14_mean
    ]]
    y = float(model.predict(x)[0])
    return {"prediction_sales": y}
