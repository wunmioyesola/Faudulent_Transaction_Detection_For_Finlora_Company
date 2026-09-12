from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import mlflow
import mlflow.sklearn
import pandas as pd

from src.interfence.prediction import (load_historical_data,
                                       predict_transaction,
                                       engineer_feature,
                                       add_transaction_to_cache,
                                       calculate_amount_usd)

from src.utility.model_loader import load_registered_model

app = FastAPI(title="fraud detection API")

model = None
historical_data = None


@app.on_event("startup")
aysnc def startup_event():
   global model, historical_data
   try:
      model = load_registered_model()
      print("model loaded successfully")
   except Exception as e:
        print(f"error occured while loading the model {e})
        model = None

    try:
        csv_path = r"C:/Users/USER/Desktop/AMDARI/Fraudulent_Transaction_Detection_For_Finlora_Company/Fraudulent_Transaction_Detection_For_Finlora_Company/fraudulent_transaction_detection.egg-info"

        historical_data = load_historical_data(csv_path)
        print("historical ata has been successfully created")
    except Exception as e:
        print("error occurred during loading of historical dataset {e}")
        historical_data = None


class Transaction(BaseModel):
    timestamp: str
    customer_id: str
    home_country: str
    sorce_currency: str
    dest_currency: str
    channel: str
    amount_src: float
    fee: float
    new_device: Optional[str] = "No"
    ip_country: str
    location_mismatch: Optional[str] = "No"
    ip_risk_score: float
    kyc_tier: str
    account_age_days: int
    device_trust_score: float
    risk_score_internal: float
    corridor_risk: float

class PredictionResponse(BaseModel):
    is_fraud: int
    fraud_probability: float
    txn_velocity_1h: Optional[int] = None
    txn_velocity_24h: Optional[int] = None
    velocity_spike: Optional[int] = None
    amount_usd: Optional[float] = None

