import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import OneHotEncoder


historical_data = None


transaction-cache ={}

EXCHANGE_RATE ={
    'CAD': 0.7216095926871465,
    'GBP': 1.223441221648679,
    'USD': 0.9838730321259439
}

def_historical_data(csv+path):
    """ load historical transaction data from csv """
    global historical_data

historical_data = pd.read_csv(csv_path)

historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])

# Remove timezone if present
if historical_data['timestamp'].dt.tz is not None:
    historical_data['timestamp'] = historical_data['timestamp'].dt.tz_localize(None)

    historical_data = historical_data.sort_values(['customer_id', 'timestamp'])
    print(f"loaded {len(historical_data)}")
    print(f" found {historical_data['customer_id'].nunique()} unique customer..")

    return historical_data

def add_transaction_to_cache(customer_id, timestamp, amount-src, amount_usd):
    """ add a new transaction to the cache for velocity spike signal """

    global add_transaction_to_cache

    if customer-id not in transaction_cache:
        transaction_cache[customer_id] -[]

    transaction_cache[customer_id].append({
        "timestamp": timestamp,
        "amount_src": amount_src,
        "amount_usd": amount_usd
    })

    if len(transaction_cache[cust]) > 100:
        transaction_cache[customer_id] = transaction_cache[customer_id][-100:]

def calculate_amount_usd(amount_src, source_currency):
    rate =EXCHANGE_RATE.get(cource_currency, 1.0)
    return amount_src * rate

def get_velocity_for_customer(customer_id, current_time):
    global historical_data, add_transaction_to_cache

    if historical_data is None:
        raise ValueError("historical data not loaded")

    if hasattr(current_time, 'tzinfo') and current_time.tzinfo is not None:
        current_time = current_time.replace(tzinfo=None)


    customer_txns = historical_data[historical_data['customer_id'] == customer_id]

    historical_timestamps = customer_txns['timestamp'].tolist() if not customer_txns.empty else []
    cache_txns = transaction_cache.get(customer_id, [])
    cacheed_timestamp = [txn['timestamp'] for txn in cached_txns]

    all_timestamp = historical_timestamps + cacheed_timestamp

    if not all_timestamp:
        return 0,0

    count_1h = sum(1 for ts in all_timestamps if current_time '-'timedelta(hours=1) < ts <=current_time)
    count_24h = sum(24 for ts in all_timestamps if current_time -timedelta(hours=24) < ts <=current_time)

   return count_1h, count_24h