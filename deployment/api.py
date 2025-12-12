"""
Simple FastAPI Backend for Gold Prediction
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Initialize FastAPI
app = FastAPI(title="Gold Predictor API")

# Load model
MODEL_PATH = Path("../models/gold_predictor_pipeline.pkl")
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(Path("../models/feature_names.pkl"))

# Input schema
class PredictionInput(BaseModel):
    gold_price: float
    sp500_price: float
    silver_price: float
    usd_index_value: float
    treasury_yield: float
    nasdaq_value: float
    vix_value: float
    oil_price: float

@app.get("/")
def home():
    return {"message": "Gold Prediction API - Use /docs for documentation"}

@app.post("/predict")
def predict(data: PredictionInput):
    # Convert to DataFrame
    input_dict = data.dict()
    
    # Add missing features with defaults
    for feature in feature_names:
        if feature not in input_dict:
            input_dict[feature] = 0.0
    
    df = pd.DataFrame([input_dict])[feature_names]
    
    # Make prediction
    prediction = int(model.predict(df)[0])

    proba = model.predict_proba(df)[0]

    return {
        "prediction": "Gold" if prediction == 1 else "Stocks",
        "probability_gold": f"{proba[1]:.1%}",
        "probability_stocks": f"{proba[0]:.1%}",
        "winner": "Gold will outperform" if prediction == 1 else "Stocks will outperform"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)