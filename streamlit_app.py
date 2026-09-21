import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path


st.set_page_config(
    page_title="Finlora Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("Finlora Fraud Detection Dashboard")
st.write("Welcome! Dashboard is up running.")


API_URL = "http://127.0.0.1:8000"

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Finlora_Dataset" / "artifacts" / "Cleaned_Data.csv"


#CSV_PATH = r"C:\Fraudulent_Transaction_Detection_For_Finlora_Company\Fraudulent_Transaction_Detection_For_Finlora_Company\Finlora_Dataset\artifacts\Cleaned_Data.csv"


@st.cache_data
def load_historical_data():
    csv_file = Path(CSV_PATH)

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Cleaned_Data.csv was not found at: {CSV_PATH}"
        )

    df = pd.read_csv(csv_file)

    return df


def find_column(df, possible_names):
    columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:
        name_lower = name.strip().lower()

        if name_lower in columns:
            return columns[name_lower]

    return None


def get_unique_values(df, possible_names):
    column = find_column(df, possible_names)

    if column is None:
        return []

    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return sorted(values)


def get_customer_data(df, customer_id):
    customer_column = find_column(
        df,
        ["customer_id"]
    )

    if customer_column is None:
        return pd.DataFrame()

    customer_rows = df[
        df[customer_column].astype(str).str.strip()
        == str(customer_id).strip()
    ]

    return customer_rows


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default

        return float(value)

    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default

        return int(float(value))

    except (ValueError, TypeError):
        return default


st.title("Finlora Fraud Detection Dashboard")

st.write(
    "Enter or select customer transaction information below "
    "to detect potentially fraudulent transactions."
)


try:
    historical_data = load_historical_data()

except Exception as error:
    st.error(
        f"Unable to load Cleaned_Data.csv: {error}"
    )

    st.stop()


customer_column = find_column(
    historical_data,
    ["customer_id"]
)


if customer_column is None:
    st.error(
        "The Cleaned_Data.csv file does not contain a customer_id column."
    )

    st.stop()


customer_ids = (
    historical_data[customer_column]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)


customer_ids = sorted(customer_ids)


st.header("Customer Information")


selected_customer = st.selectbox(
    "Customer ID",
    customer_ids
)


customer_data = get_customer_data(
    historical_data,
    selected_customer
)


if customer_data.empty:
    st.warning(
        "No historical information was found for this customer."
    )

    st.stop()


latest_customer_data = customer_data.iloc[-1]


st.subheader("Preloaded Customer Details")


col1, col2, col3 = st.columns(3)


home_country_column = find_column(
    historical_data,
    ["home_country"]
)


if home_country_column:

    home_country_values = get_unique_values(
        historical_data,
        ["home_country"]
    )

    customer_home_country = str(
        latest_customer_data[home_country_column]
    ).strip()

    if customer_home_country not in home_country_values:
        home_country_values.insert(
            0,
            customer_home_country
        )

    selected_home_country = col1.selectbox(
        "Home Country",
        home_country_values,
        index=home_country_values.index(
            customer_home_country
        )
    )

else:

    selected_home_country = col1.text_input(
        "Home Country"
    )


kyc_column = find_column(
    historical_data,
    ["kyc_tier"]
)


if kyc_column:

    kyc_values = get_unique_values(
        historical_data,
        ["kyc_tier"]
    )

    customer_kyc = str(
        latest_customer_data[kyc_column]
    ).strip()

    if customer_kyc not in kyc_values:
        kyc_values.insert(
            0,
            customer_kyc
        )

    selected_kyc_tier = col2.selectbox(
        "KYC Tier",
        kyc_values,
        index=kyc_values.index(
            customer_kyc
        )
    )

else:

    selected_kyc_tier = col2.text_input(
        "KYC Tier"
    )


account_age_column = find_column(
    historical_data,
    ["account_age_days"]
)


if account_age_column:

    account_age_value = safe_int(
        latest_customer_data[account_age_column]
    )

else:

    account_age_value = 0


selected_account_age = col3.number_input(
    "Account Age (Days)",
    min_value=0,
    value=account_age_value,
    step=1
)


st.header("Device and Risk Information")


col1, col2, col3 = st.columns(3)


device_trust_column = find_column(
    historical_data,
    ["device_trust_score"]
)


if device_trust_column:

    device_trust_value = safe_float(
        latest_customer_data[device_trust_column]
    )

else:

    device_trust_value = 0.0


selected_device_trust = col1.number_input(
    "Device Trust Score",
    min_value=0.0,
    value=device_trust_value,
    step=0.01
)


risk_score_column = find_column(
    historical_data,
    ["risk_score_internal"]
)


if risk_score_column:

    risk_score_value = safe_float(
        latest_customer_data[risk_score_column]
    )

else:

    risk_score_value = 0.0


selected_risk_score = col2.number_input(
    "Internal Risk Score",
    min_value=0.0,
    value=risk_score_value,
    step=0.01
)


corridor_risk_column = find_column(
    historical_data,
    ["corridor_risk"]
)


if corridor_risk_column:

    corridor_risk_value = safe_float(
        latest_customer_data[corridor_risk_column]
    )

else:

    corridor_risk_value = 0.0


selected_corridor_risk = col3.number_input(
    "Corridor Risk",
    min_value=0.0,
    value=corridor_risk_value,
    step=0.01
)


st.header("Transaction Information")


col1, col2, col3 = st.columns(3)


source_currency_values = get_unique_values(
    historical_data,
    ["source_currency"]
)


source_currency_column = find_column(
    historical_data,
    ["source_currency"]
)


if source_currency_column:

    customer_source_currency = str(
        latest_customer_data[source_currency_column]
    ).strip()

else:

    customer_source_currency = ""


if customer_source_currency:
    if customer_source_currency not in source_currency_values:
        source_currency_values.insert(
            0,
            customer_source_currency
        )


if source_currency_values:

    selected_source_currency = col1.selectbox(
        "Source Currency",
        source_currency_values
    )

else:

    selected_source_currency = col1.text_input(
        "Source Currency"
    )


dest_currency_values = get_unique_values(
    historical_data,
    ["dest_currency"]
)


dest_currency_column = find_column(
    historical_data,
    ["dest_currency"]
)


if dest_currency_column:

    customer_dest_currency = str(
        latest_customer_data[dest_currency_column]
    ).strip()

else:

    customer_dest_currency = ""


if customer_dest_currency:
    if customer_dest_currency not in dest_currency_values:
        dest_currency_values.insert(
            0,
            customer_dest_currency
        )


if dest_currency_values:

    selected_dest_currency = col2.selectbox(
        "Destination Currency",
        dest_currency_values
    )

else:

    selected_dest_currency = col2.text_input(
        "Destination Currency"
    )


channel_values = get_unique_values(
    historical_data,
    ["channel", "transaction_channel"]
)


channel_column = find_column(
    historical_data,
    ["channel", "transaction_channel"]
)


if channel_column:

    customer_channel = str(
        latest_customer_data[channel_column]
    ).strip()

else:

    customer_channel = ""


if customer_channel:
    if customer_channel not in channel_values:
        channel_values.insert(
            0,
            customer_channel
        )


if channel_values:

    selected_channel = col3.selectbox(
        "Transaction Channel",
        channel_values
    )

else:

    selected_channel = col3.text_input(
        "Transaction Channel"
    )


col1, col2, col3 = st.columns(3)


amount_column = find_column(
    historical_data,
    ["amount_src"]
)


if amount_column:

    amount_value = safe_float(
        latest_customer_data[amount_column]
    )

else:

    amount_value = 0.0


selected_amount = col1.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=amount_value,
    step=0.01
)


fee_column = find_column(
    historical_data,
    ["fee"]
)


if fee_column:

    fee_value = safe_float(
        latest_customer_data[fee_column]
    )

else:

    fee_value = 0.0


selected_fee = col2.number_input(
    "Transaction Fee",
    min_value=0.0,
    value=fee_value,
    step=0.01
)


selected_timestamp = col3.text_input(
    "Transaction Timestamp",
    value=datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)


st.header("Location and IP Information")


col1, col2, col3 = st.columns(3)


ip_country_values = get_unique_values(
    historical_data,
    ["ip_country"]
)


ip_country_column = find_column(
    historical_data,
    ["ip_country"]
)


if ip_country_column:

    customer_ip_country = str(
        latest_customer_data[ip_country_column]
    ).strip()

else:

    customer_ip_country = ""


if customer_ip_country:
    if customer_ip_country not in ip_country_values:
        ip_country_values.insert(
            0,
            customer_ip_country
        )


if ip_country_values:

    selected_ip_country = col1.selectbox(
        "IP Country",
        ip_country_values
    )

else:

    selected_ip_country = col1.text_input(
        "IP Country"
    )


ip_risk_column = find_column(
    historical_data,
    ["ip_risk_score"]
)


if ip_risk_column:

    ip_risk_value = safe_float(
        latest_customer_data[ip_risk_column]
    )

else:

    ip_risk_value = 0.0


selected_ip_risk = col2.number_input(
    "IP Risk Score",
    min_value=0.0,
    value=ip_risk_value,
    step=0.01
)


location_mismatch_values = get_unique_values(
    historical_data,
    ["location_mismatch"]
)


location_mismatch_column = find_column(
    historical_data,
    ["location_mismatch"]
)


if location_mismatch_column:

    customer_location_mismatch = str(
        latest_customer_data[location_mismatch_column]
    ).strip()

else:

    customer_location_mismatch = ""


if customer_location_mismatch:
    if customer_location_mismatch not in location_mismatch_values:
        location_mismatch_values.insert(
            0,
            customer_location_mismatch
        )


if location_mismatch_values:

    selected_location_mismatch = col3.selectbox(
        "Location Mismatch",
        location_mismatch_values
    )

else:

    selected_location_mismatch = col3.text_input(
        "Location Mismatch",
        value="No"
    )


st.header("Device Status")


new_device_values = get_unique_values(
    historical_data,
    ["new_device"]
)


new_device_column = find_column(
    historical_data,
    ["new_device"]
)


if new_device_column:

    customer_new_device = str(
        latest_customer_data[new_device_column]
    ).strip()

else:

    customer_new_device = ""


if customer_new_device:
    if customer_new_device not in new_device_values:
        new_device_values.insert(
            0,
            customer_new_device
        )


if new_device_values:

    selected_new_device = st.selectbox(
        "New Device",
        new_device_values
    )

else:

    selected_new_device = st.text_input(
        "New Device",
        value="No"
    )


transaction_payload = {
    "timestamp": selected_timestamp,
    "customer_id": str(selected_customer),
    "home_country": str(selected_home_country),
    "source_currency": str(selected_source_currency),
    "dest_currency": str(selected_dest_currency),
    "channel": str(selected_channel),
    "amount_src": float(selected_amount),
    "fee": float(selected_fee),
    "new_device": str(selected_new_device),
    "ip_country": str(selected_ip_country),
    "location_mismatch": str(selected_location_mismatch),
    "ip_risk_score": float(selected_ip_risk),
    "kyc_tier": str(selected_kyc_tier),
    "account_age_days": int(selected_account_age),
    "device_trust_score": float(selected_device_trust),
    "risk_score_internal": float(selected_risk_score),
    "corridor_risk": float(selected_corridor_risk)
}


with st.expander("View Transaction Data Sent to API"):
    st.json(transaction_payload)


st.header("Fraud Detection")


if st.button(
    "Run Fraud Detection",
    type="primary",
    use_container_width=True
):

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=10
        )


        if health_response.status_code != 200:

            st.error(
                "The FastAPI server is not responding correctly."
            )

            st.stop()


        health_data = health_response.json()


        if not health_data.get("model_loaded", False):

            st.error(
                "The API is running, but the ML model is not loaded."
            )

            st.stop()


        if not health_data.get("historical_data_loaded", False):

            st.error(
                "The API is running, but the historical dataset "
                "has not been loaded."
            )

            st.stop()


        with st.spinner(
            "Running fraud detection..."
        ):

            response = requests.post(
                f"{API_URL}/predict",
                json=transaction_payload,
                timeout=30
            )


        if response.status_code == 200:

            result = response.json()


            is_fraud = int(
                result.get(
                    "is_fraud",
                    0
                )
            )


            fraud_probability = float(
                result.get(
                    "fraud_probability",
                    0
                )
            )


            txn_velocity_1h = result.get(
                "txn_velocity_1h"
            )


            txn_velocity_24h = result.get(
                "txn_velocity_24h"
            )


            velocity_spike = result.get(
                "velocity_spike"
            )


            amount_usd = result.get(
                "amount_usd"
            )


            if is_fraud == 1:

                st.error(
                    "🚨 FRAUDULENT TRANSACTION DETECTED"
                )

            else:

                st.success(
                    "✅ TRANSACTION NOT FLAGGED AS FRAUD"
                )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Prediction",
                    "Fraud" if is_fraud == 1
                    else "Not Fraud"
                )


            with col2:

                st.metric(
                    "Fraud Probability",
                    f"{fraud_probability:.2%}"
                )


            with col3:

                if amount_usd is not None:

                    st.metric(
                        "Amount USD",
                        f"${float(amount_usd):,.2f}"
                    )

                else:

                    st.metric(
                        "Amount USD",
                        "N/A"
                    )


            with col4:

                if velocity_spike is not None:

                    st.metric(
                        "Velocity Spike",
                        str(velocity_spike)
                    )

                else:

                    st.metric(
                        "Velocity Spike",
                        "N/A"
                    )


            st.subheader(
                "Transaction Velocity"
            )


            col1, col2 = st.columns(2)


            with col1:

                if txn_velocity_1h is not None:

                    st.metric(
                        "Transactions in 1 Hour",
                        txn_velocity_1h
                    )

                else:

                    st.metric(
                        "Transactions in 1 Hour",
                        "N/A"
                    )


            with col2:

                if txn_velocity_24h is not None:

                    st.metric(
                        "Transactions in 24 Hours",
                        txn_velocity_24h
                    )

                else:

                    st.metric(
                        "Transactions in 24 Hours",
                        "N/A"
                    )


            with st.expander(
                "View Complete API Response"
            ):

                st.json(result)


        else:

            st.error(
                f"API returned status code {response.status_code}"
            )


            try:

                error_data = response.json()

                st.json(error_data)

            except Exception:

                st.code(
                    response.text
                )


    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the FastAPI server."
        )

        st.info(
            "Make sure FastAPI is running at http://127.0.0.1:8000"
        )


    except requests.exceptions.Timeout:

        st.error(
            "The request to the FastAPI server timed out."
        )


    except Exception as error:

        st.error(
            f"An unexpected error occurred: {error}"
        )

