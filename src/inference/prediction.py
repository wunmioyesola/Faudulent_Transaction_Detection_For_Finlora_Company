import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import OneHotEncoder


historical_data = None


transaction_cache ={}

EXCHANGE_RATE = {
    'CAD': 0.7216095926871465,
    'GBP': 1.223441221648679,
    'USD': 0.9838730321259439
}

def load_historical_data(csv_path):
    """ load historical transaction data from csv """
    global historical_data

    historical_data = pd.read_csv(csv_path)

    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])

# Remove timezone if present
    if  historical_data['timestamp'].dt.tz is not None:
        historical_data['timestamp'] = historical_data['timestamp'].dt.tz_localize(None)

    historical_data = historical_data.sort_values(['customer_id', 'timestamp'])
    print(f"loaded {len(historical_data)}")
    print(f" found {historical_data['customer_id'].nunique()} unique customer..")

    return historical_data

def add_transaction_to_cache(customer_id, timestamp, amount_src, amount_usd):
    """ add a new transaction to the cache for velocity spike signal """

    global transaction_cache

    if customer_id not in transaction_cache:
        transaction_cache[customer_id] = []

    transaction_cache[customer_id].append({
        "timestamp": timestamp,
        "amount_src": amount_src,
        "amount_usd": amount_usd
    })

    if len(transaction_cache[customer_id]) > 100:
        transaction_cache[customer_id] = transaction_cache[customer_id][-100:]

def calculate_amount_usd(amount_src, source_currency):
    rate =EXCHANGE_RATE.get(source_currency, 1.0)
    return amount_src * rate

    #return float(amount_src) * rate

def get_velocity_for_customer(customer_id, current_time):
    global historical_data, add_transaction_cache

    if historical_data is None:
        raise ValueError("historical data not loaded")

    if hasattr(current_time, 'tzinfo') and current_time.tzinfo is not None:
        current_time = current_time.replace(tzinfo=None)


    customer_txns = historical_data[historical_data['customer_id'] == customer_id]

    historical_timestamps = customer_txns['timestamp'].tolist() if not customer_txns.empty else []
    cache_txns = transaction_cache.get(customer_id, [])
    cached_timestamp = [txn['timestamp'] for txn in cache_txns]

    all_timestamps = historical_timestamps + cached_timestamp

    if not all_timestamps:
        return 0,0

    count_1h = sum(1 for ts in all_timestamps if current_time - timedelta(hours=1) < ts <=current_time)
    count_24h = sum(1 for ts in all_timestamps if current_time - timedelta(hours=24) < ts <=current_time)

    return count_1h, count_24h

def engineer_feature(data):
        global historical_data
        df = data.copy()

        #numeric_cols = ['amount_usd', 'ip_risk_score', 'device_trust_score', 'risk_score_internal', 'corridor_risk', 'account_age_days']
        #for col in numeric_cols:
            #if col in df.columns:
                #df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        if 'amount_usd' not in df.columns or df['amount_usd'].isna().sum():
            df['amount_usd'] = df.apply(
                lambda row: calculate_amount_usd(row['amount_src'], row['source_currency']),
                axis=1
            )

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >=5).astype(int)


        # calculating the velocity spike for a particular user
        #if 'customer_id' in df.columns and len(df) > 0:
        #if 'customer_id' in df.columns:
            #fx_1h = []
            #fx_24h = []
            #for customer_id, txn_time in zip(df['customer_id'], df['timestamp']):
                #count_1h, count_24 = get_velocity_for_customer(customer_id, txn_time)
                #fx_1h.append(count_1h)
                #fx_24h.append(count_24)
                
            #df['txn_velocity_1h'] = fx_1h
            #df['txn_velocity_24h'] = fx_24h

        #print("Velocity calculation complete")

        # calculating the velocity spike for a particular user
            #print("Calculating velocity for customers...")
        if 'customer_id' in df.columns:
            for idx, row in df.iterrows():
                customer_id = row['customer_id']
                txn_time = row['timestamp']

                # get velocity for the user
                count_1h, count_24h = get_velocity_for_customer(customer_id, txn_time)

                df.loc[idx, 'txn_velocity_1h'] = count_1h
                df.loc[idx, 'txn_velocity_24h'] = count_24h
        #print("Velocity calculation complete")

        # Creating a threshold based features from the following risk_signal
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['late_night_hours'] = ((df['hour'] >= 3) & (df['hour'] <= 7)).astype(int)
        df['amount_high'] = (df['amount_usd'] >= 1000).astype(int)
        df['high_ip_risk'] = (df['ip_risk_score'] > 0.8).astype(int)
        df['low_device_trust'] = (df['device_trust_score'] > 0.5).astype(int)
        df['new_account'] = ((df['account_age_days'] >= 30) & (df['account_age_days'] < 90)).astype(int)
        df['very_new_account'] = (df['account_age_days'] < 30).astype(int)
        df['velocity_spike'] = (df['txn_velocity_1h'] >= 3).astype(int)

        return df

def predict_transaction(model, input_data):
    """ 
    make predictions
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    df_engineered = engineer_feature(df)
    print(df_engineered)
    
    prediction = model.predict(df_engineered)[0]
    prediction_probability = model.predict_proba(df_engineered)[0][1]
    
    return prediction, prediction_probability

    #for col in df.select_dtypes(include=['object', 'category']).columns:
            #if col not in ['ip_country', 'transaction_channel']:
                #df[col] = df[col].astype(str).str.strip() #.str.lower()  
                
    #if 'source_currency' in df.columns:
        #df['source_currency'] = df['source_currency'].astype(str).str.strip().str.upper()

    #if 'destination_currency' in df.columns:
            #df['destination_currency'] = df['destination_currency'].astype(str).str.strip().str.upper() 

    #if 'ip_country' in df.columns:
            #df['ip_country'] = df['ip_country'].astype(str).str.strip().str.lower()

    #if 'transaction_channel' in df.columns:
            #df['transaction_channel'] = df['transaction_channel'].astype(str).str.strip().str.lower()

    #for col in df.select_dtypes(include=['object', 'category']).columns:
        #if col in df.columns:
            #df[col] = df[col].astype(str).str.upper()

    # Normalize 'new_device' to lowercase to avoid unknown category errors
    #if 'new_device' in df.columns:
        #df['new_device'] = df['new_device'].map({'no': 0, 'yes': 1}).fillna(0)
        #df['new_device'] = df['new_device'].fillna(0)
        #df['new_device'] = df['new_device'].map({'No': 0, 'Yes': 1, 'no': 0, 'yes':1})

    df_engineered = engineer_feature(df)
    #print(df_engineered)

    #prediction = model.predict(df_engineered)[0]
    #prediction_probability = model.predict_proba(df_engineered)[0][1]

    #return prediction, prediction_probability