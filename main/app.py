#import sys
#from pathlib import Path
#sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional
import mlflow
import mlflow.sklearn
import pandas as pd

from src.inference.prediction import (load_historical_data,
                                      predict_transaction,
                                      engineer_feature,
                                      add_transaction_to_cache,
                                      calculate_amount_usd)

from src.utility.model_loader import load_registered_model

app = FastAPI(title="fraud detection API")

model = None
historical_data = None


@app.on_event("startup")
async def startup_event():
   global model, historical_data
   try:
      model = load_registered_model()
      print("model loaded successfully")
   except Exception as e:
        print(f"error occured while loading the model {e}")
        model = None

try:
        #csv_path = Path(__file__).resolve().parent.parent / "Finlora Dataset" / "artifacts" / "Cleaned_Data.csv"
        csv_path = r"C:/Fraudulent_Transaction_Detection_For_Finlora_Company/Fraudulent_Transaction_Detection_For_Finlora_Company/Finlora Dataset/artifacts/Cleaned_Data.csv"

        historical_data = load_historical_data(csv_path)
        print("historical data has been successfully created")
except Exception as e:
        print("error occurred during loading of historical dataset {e}")
        historical_data = None

        #print(f"CRITICAL ERROR during startup: {e}")
        #raise e
        #historical_data = None

# what the API is expecting from the user as an input
class TransactionData(BaseModel):
    timestamp: str
    customer_id: str
    home_country: str
    source_currency: str
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

# what the API is going to giving back to the user as an output or response
class PredictionResponse(BaseModel):
    is_fraud: int
    fraud_probability: float
    txn_velocity_1h: Optional[int] = None
    txn_velocity_24h: Optional[int] = None
    velocity_spike: Optional[int] = None
    amount_usd: Optional[float] = None

# crating the predict API so that our model can get users requests and make prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionData):
    global model, historical_data

    if model is None:
        raise HTTPException(status_code = 500, detail="model not loaded")

    if historical_data is None:
        raise HTTPException(status_code =500, detail = "historical data has not been loaded")

    try:
        input_data = transaction.dict()

        prediction, prediction_probability = predict_transaction(model, input_data)

        #input_data = jsonable_encoder(transaction)
        #input_data = {
            #k: (v.lower() if isinstance(v, str) else v)
            #for k, v in input_data.items()
        #}
        #for key, value in input_data.items():
            #if isinstance(value, str):
                #input_data[key] = value.lower()
       
    # Feature engineering for the response
        df = pd.DataFrame([input_data])
        df_engineered = engineer_feature(df) 
        
        amount_usd = calculate_amount_usd(transaction.amount_src, transaction.source_currency)

        # cache metrics
        add_transaction_to_cache(
            transaction.customer_id,
            pd.to_datetime(transaction.timestamp),
            float(transaction.amount_src),
            amount_usd
        )

        return PredictionResponse(
            is_fraud = int(prediction),
            fraud_probability = float(prediction_probability),
            txn_velocity_1h = int(df_engineered.iloc[0]['txn_velocity_1h'] if 'txn_velocity_1h' in df_engineered.columns else None),
            txn_velocity_24h = int(df_engineered.iloc[0]['txn_velocity_24h'] if 'txn_velocity_24h' in df_engineered.columns else None),
            velocity_spike = int(df_engineered.iloc[0]['velocity_spike'] if 'velocity_spike' in df_engineered.columns else None),
            amount_usd = float(df_engineered.iloc[0]['amount_usd']) if 'amount_usd' in df_engineered.columns else None
        )
            
    except Exception as e:
        raise HTTPException(status_code=400, detail = str(e))

        for col in df.columns:
            #if df[col].dtypes == object or str(df[col].dtype) == 'string':
                df[col] = df[col].astype(str).str.lower()

        df_engineered = engineer_feature(df) 

        for col in df_engineered.select_dtypes(include=['object']).columns:
            df_engineered[col] = df_engineered[col].astype(str).str.lower()

        prediction, prediction_probability = predict_transaction(model, df_engineered)
        
        #return {
            #"prediction": prediction,
            #"probability": prediction_probability
        #}
       
@app.get("/health")
async def health_check():
    return {
        "status": "health",
        "model_loaded": model is not None,
        "historical_data_loaded": historical_data is not None
    }