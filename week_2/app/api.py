from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd


app = FastAPI(title="Churn Model API (Isotonic Calibration)")

# -------------------------------
# Input schema for prediction
# Only features used in training (exclude TotalCharges & customerID)
# -------------------------------
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float

# -------------------------------
# Prediction endpoint
# -------------------------------
pipeline = joblib.load("final_churn_model_calibrated_isotonic.pkl")
THRESHOLD = 0.1275  # <-- use cost-minimizing threshold

@app.post("/predict")
def predict(customer: CustomerInput):
    # Convert Pydantic model to single-row DataFrame
    df_input = pd.DataFrame([customer.dict()])
    
    # Predict probability
    prob = pipeline.predict_proba(df_input)[0][1]
    pred = int(prob >= THRESHOLD)
    
    return {"probability": float(prob), "prediction": pred, "threshold": float(THRESHOLD)}
# Health check endpoint
# -------------------------------
@app.get("/")
def health():
    return {"status": "Isotonic-Calibrated Churn Model API running"}
