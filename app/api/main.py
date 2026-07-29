from __future__ import annotations 
 
from datetime import datetime, timezone 
from pathlib import Path 
from uuid import uuid4 
 
import joblib 
import pandas as pd 
from fastapi import FastAPI 
from pydantic import BaseModel, Field 
 
MODEL_PATH = "models/model.joblib" 
artifact = joblib.load(MODEL_PATH) 
model = artifact["pipeline"] 
threshold = float(artifact["threshold"]) 
app = FastAPI(title="Riyadh Air Quality API", version="1.0.0") 
 
 
class PredictionRequest(BaseModel): 
    pm2_5: float = Field(ge=0, le=1000) 
    pm10: float = Field(ge=0, le=1500) 
    temperature_2m: float = Field(ge=-20, le=65) 
    relative_humidity_2m: float = Field(ge=0, le=100) 
    wind_speed_10m: float = Field(ge=0, le=200) 
    hour: int = Field(ge=0, le=23) 
    day_of_week: int = Field(ge=0, le=6) 
    pm2_5_lag_1: float = Field(ge=0, le=1000) 
    pm2_5_lag_3: float = Field(ge=0, le=1000) 
    pm2_5_rolling_mean_6: float = Field(ge=0, le=1000) 
 
 
@app.get("/health") 
def health() -> dict[str, object]: 
    return {"status": "ok", "model_loaded": model is not None} 
 
 
@app.get("/model-info") 
def model_info() -> dict[str, object]: 
    return {"version": "1.0.0", "threshold": threshold} 
 
 
@app.post("/predict") 
def predict(request: PredictionRequest) -> dict[str, object]: 
    request_id = str(uuid4()) 
    frame = pd.DataFrame([request.model_dump()]) 
    probability = float(model.predict_proba(frame)[:, 1][0]) 
    prediction = int(probability >= threshold) 
 
    log_row = frame.copy() 
    log_row["probability"] = probability 
    log_row["prediction"] = prediction 
    log_row["timestamp"] = datetime.now(timezone.utc).isoformat() 
    log_path = Path("data/monitoring/predictions.csv") 
    log_path.parent.mkdir(parents=True, exist_ok=True) 
    log_row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False) 
 
    return { 
        "request_id": request_id, 
        "prediction": prediction, 
        "probability": probability, 
        "risk_level": "high" if prediction else "normal", 
        "model_version": "1.0.0", 
    } 