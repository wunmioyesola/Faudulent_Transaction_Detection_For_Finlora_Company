import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Finlora Fraud Detection Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CSV_PATH = "Finlora Dataset/artifacts/Cleaned_Data.csv"

DEFAULT_API_URL = "http://127.0.0.1:8000"

CSV_PATH = os.getenv(
    "FINLORA_CSV_PATH",
    DEFAULT_CSV_PATH
)

API_URL = os.getenv(
    "FINLORA_API_URL",
    DEFAULT_API_URL
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

        .main-title {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 16px;
            color: #666666;
            margin-bottom: 25px;
        }

        .section-title {
            font-size: 21px;
            font-weight: 600;
            margin-top: 15px;
            margin-bottom: 10px;
        }

        .result-card {
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #dddddd;
            margin-top: 15px;
        }

        .metric-label {
            font-size: 14px;
            color: #666666;
        }

        .metric-value {
            font-size: 22px;
            font-weight: 600;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD CLEANED HISTORICAL DATA
# ============================================================

@st.cache_data
def load_data(file_path):

    data = pd.read_csv(file_path)

    if "timestamp" in data.columns:

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce"
        )

    return data


try:

    historical_data = load_data(
        CSV_PATH
    )

except FileNotFoundError:

    st.error(
        "The Cleaned_Data.csv file could not be found.\n\n"
        f"Please check the following path:\n{CSV_PATH}"
    )

    st.stop()

except Exception as e:

    st.error(
        f"An error occurred while loading "
        f"Cleaned_Data.csv: {e}"
    )

    st.stop()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "customer_id",
    "timestamp",
    "home_country",
    "source_currency",
    "dest_currency",
    "channel",
    "amount_src",
    "fee",
    "new_device",
    "ip_country",
    "location_mismatch",
    "ip_risk_score",
    "kyc_tier",
    "account_age_days",
    "device_trust_score",
    "risk_score_internal",
    "corridor_risk"
]


missing_columns = [
    column
    for column in required_columns
    if column not in historical_data.columns
]


if missing_columns:

    st.error(
        "The following required columns are missing "
        "from Cleaned_Data.csv:\n\n"
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# HELPER FUNCTION - GET UNIQUE VALUES
# ============================================================

def get_unique_values(
    dataframe,
    column
):

    if column not in dataframe.columns:

        return []

    values = (
        dataframe[column]
        .dropna()
        .unique()
        .tolist()
    )

    values = sorted(
        values,
        key=lambda value: str(value).lower()
    )

    return values


# ============================================================
# HELPER FUNCTION - GET CUSTOMER HISTORY
# ============================================================

def get_customer_history(
    dataframe,
    customer_id
):

    customer_records = dataframe[
        dataframe["customer_id"].astype(str)
        == str(customer_id)
    ].copy()

    if customer_records.empty:

        return customer_records

    if "timestamp" in customer_records.columns:

        customer_records = (
            customer_records.sort_values(
                by="timestamp",
                ascending=False
            )
        )

    return customer_records


# ============================================================
# HELPER FUNCTION - GET MOST RECENT CUSTOMER RECORD
# ============================================================

def get_customer_data(
    dataframe,
    customer_id
):

    customer_records = get_customer_history(
        dataframe,
        customer_id
    )

    if customer_records.empty:

        return None

    return customer_records.iloc[0]


# ============================================================
# HELPER FUNCTION - SAFE INTEGER
# ============================================================

def safe_int(
    value,
    default=0
):

    if value is None:

        return default

    try:

        if pd.isna(value):

            return default

    except (TypeError, ValueError):

        pass

    try:

        return int(float(value))

    except (TypeError, ValueError):

        return default


# ============================================================
# HELPER FUNCTION - SAFE FLOAT
# ============================================================

def safe_float(
    value,
    default=0.0
):

    if value is None:

        return default

    try:

        if pd.isna(value):

            return default

    except (TypeError, ValueError):

        pass

    try:

        return float(value)

    except (TypeError, ValueError):

        return default


# ============================================================
# HELPER FUNCTION - SAFE STRING
# ============================================================

def safe_string(
    value,
    default=""
):

    if value is None:

        return default

    try:

        if pd.isna(value):

            return default

    except (TypeError, ValueError):

        pass

    return str(value)


# ============================================================
# HELPER FUNCTION - BOOLEAN DISPLAY
# ============================================================

def boolean_to_yes_no(value):

    if value is None:

        return "No"

    try:

        if pd.isna(value):

            return "No"

    except (TypeError, ValueError):

        pass

    if isinstance(value, bool):

        return "Yes" if value else "No"

    value_string = str(
        value
    ).strip().lower()

    if value_string in [
        "true",
        "1",
        "yes",
        "y"
    ]:

        return "Yes"

    return "No"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Finlora Fraud Detection Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Powered Fraud Transaction Analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Dashboard Settings"
    )

    api_url = st.text_input(
        "FastAPI URL",
        value=API_URL
    )

    st.divider()

    st.subheader(
        "Dataset Information"
    )

    st.write(
        f"**Transactions:** "
        f"{len(historical_data):,}"
    )

    st.write(
        f"**Customers:** "
        f"{historical_data['customer_id'].nunique():,}"
    )

    if "is_fraud" in historical_data.columns:

        fraud_values = pd.to_numeric(
            historical_data["is_fraud"],
            errors="coerce"
        ).fillna(0)

        fraud_count = int(
            fraud_values.sum()
        )

        if len(historical_data) > 0:

            fraud_rate = (
                fraud_count
                / len(historical_data)
                * 100
            )

        else:

            fraud_rate = 0.0

        st.write(
            f"**Fraud Records:** "
            f"{fraud_count:,}"
        )

        st.write(
            f"**Historical Fraud Rate:** "
            f"{fraud_rate:.2f}%"
        )

    st.divider()

    st.subheader(
        "API Status"
    )

    try:

        health_response = requests.get(
            f"{api_url.rstrip('/')}/health",
            timeout=5
        )

        if health_response.status_code == 200:

            health_data = (
                health_response.json()
            )

            if (
                health_data.get("status")
                == "healthy"
            ):

                st.success(
                    "API is online"
                )

            else:

                st.warning(
                    "API is online but "
                    "not fully ready"
                )

            model_loaded = health_data.get(
                "model_loaded",
                False
            )

            historical_loaded = (
                health_data.get(
                    "historical_data_loaded",
                    False
                )
            )

            st.write(
                "Model loaded: "
                f"{'Yes' if model_loaded else 'No'}"
            )

            st.write(
                "Historical data loaded: "
                f"{'Yes' if historical_loaded else 'No'}"
            )

        else:

            st.warning(
                "API returned HTTP "
                f"{health_response.status_code}"
            )

    except requests.exceptions.RequestException:

        st.error(
            "API is offline"
        )

    st.caption(
        "Finlora Fraud Detection System"
    )


# ============================================================
# CUSTOMER SELECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '1. Customer Selection'
    '</div>',
    unsafe_allow_html=True
)


customer_ids = (
    historical_data["customer_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

customer_ids.sort()


if not customer_ids:

    st.error(
        "No customer IDs were found "
        "in the historical dataset."
    )

    st.stop()


selected_customer = st.selectbox(
    "Select Customer ID",
    options=customer_ids
)


# ============================================================
# GET SELECTED CUSTOMER INFORMATION
# ============================================================

customer_record = get_customer_data(
    historical_data,
    selected_customer
)


if customer_record is None:

    st.warning(
        "No historical information was found "
        "for the selected customer."
    )

    st.stop()


customer_history = get_customer_history(
    historical_data,
    selected_customer
)


# ============================================================
# CUSTOMER ACCOUNT PROFILE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '2. Customer Account Profile'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "The customer account information below is "
    "automatically preloaded from the customer's "
    "most recent historical transaction."
)


profile_col1, profile_col2, profile_col3 = (
    st.columns(3)
)


# ============================================================
# PROFILE COLUMN 1
# ============================================================

with profile_col1:
    st.text_input(
        "Customer ID",
        value=safe_string(selected_customer
        ),
        disabled=False
    )

    st.number_input(
        "Acount Age (Days)",
        value=safe_int(
            customer_record[
                "account_age_days"
                ]
        ),
        disabled=False
    )

# ============================================================
# PROFILE COLUMN 2
# ============================================================

with profile_col2:

    st.text_input(
        "Home Country",
        value=safe_string(
            customer_record["home_country"],
            "Not availale"
        ),
        disabled=False
    )

    st.number_input(
        "Device Trust Score",
        value=safe_float(
            customer_record[
                "device_trust_score"
            ]
        ),
        format="%.4f",
        disabled=False
    )
  
# ============================================================
# PROFILE COLUMN 3
# ============================================================

with profile_col3:

    st.text_input(
        "KYC Tier",
        value=safe_float(
            customer_record["kyc_tier"],
            "Not availale"
        ),
        disabled=False
    )

    st.number_input(
        "Internal Risk Score",
        value=safe_float(
                   customer_record[
                       "risk_score_internal"
                   ]
        ),
        format="%.4f",
        disabled=False
    )

# ============================================================
# CUSTOMER TRANSACTION HISTORY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '3. Customer Transaction History'
    '</div>',
    unsafe_allow_html=True
)


history_col1, history_col2, history_col3 = (
    st.columns(3)
)


# ============================================================
# TOTAL TRANSACTIONS
# ============================================================

with history_col1:

    st.metric(
        "Total Transactions",
        f"{len(customer_history):,}"
    )


# ============================================================
# HISTORICAL FRAUD CASES
# ============================================================

with history_col2:

    if "is_fraud" in customer_history.columns:

        historical_fraud_values = (
            pd.to_numeric(
                customer_history[
                    "is_fraud"
                ],
                errors="coerce"
            )
            .fillna(0)
        )

        historical_fraud_count = int(
            historical_fraud_values.sum()
        )

    else:

        historical_fraud_count = 0

    st.metric(
        "Historical Fraud Cases",
        f"{historical_fraud_count:,}"
    )


# ============================================================
# TOTAL TRANSACTION VALUE
# ============================================================

with history_col3:

    if "amount_usd" in customer_history.columns:

        amount_values = pd.to_numeric(
            customer_history[
                "amount_usd"
            ],
            errors="coerce"
        ).fillna(0)

        total_amount_usd = (
            amount_values.sum()
        )

    else:

        total_amount_usd = 0.0

    st.metric(
        "Total Transaction Value",
        f"${total_amount_usd:,.2f}"
    )


# ============================================================
# CUSTOMER HISTORY TABLE
# ============================================================

with st.expander(
    "View Customer Transaction History"
):

    history_display_columns = [
        "transaction_id",
        "timestamp",
        "source_currency",
        "dest_currency",
        "channel",
        "amount_src",
        "amount_usd",
        "fee",
        "new_device",
        "ip_country",
        "location_mismatch",
        "ip_risk_score",
        "is_fraud"
    ]

    available_history_columns = [
        column
        for column in history_display_columns
        if column in customer_history.columns
    ]

    st.dataframe(
        customer_history[
            available_history_columns
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# NEW TRANSACTION DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '4. New Transaction Details'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the details of the new transaction. "
    "Categorical options are automatically loaded "
    "from Cleaned_Data.csv."
)


# ============================================================
# DYNAMIC CATEGORICAL VALUES FROM CSV
# ============================================================

source_currency_values = get_unique_values(
    historical_data,
    "source_currency"
)

dest_currency_values = get_unique_values(
    historical_data,
    "dest_currency"
)

channel_values = get_unique_values(
    historical_data,
    "channel"
)

ip_country_values = get_unique_values(
    historical_data,
    "ip_country"
)


# ============================================================
# DYNAMIC BOOLEAN VALUES FROM CSV
# ============================================================

new_device_values = sorted(
    list(
        set(
            boolean_to_yes_no(value)
            for value in historical_data[
                "new_device"
            ]
            .dropna()
            .unique()
        )
    )
)


location_mismatch_values = sorted(
    list(
        set(
            boolean_to_yes_no(value)
            for value in historical_data[
                "location_mismatch"
            ]
            .dropna()
            .unique()
        )
    )
)


# ============================================================
# CHECK CATEGORICAL VALUES
# ============================================================

if not source_currency_values:

    st.error(
        "No source currency values were found "
        "in the historical dataset."
    )

    st.stop()


if not dest_currency_values:

    st.error(
        "No destination currency values were found "
        "in the historical dataset."
    )

    st.stop()


if not channel_values:

    st.error(
        "No transaction channel values were found "
        "in the historical dataset."
    )

    st.stop()


if not ip_country_values:

    st.error(
        "No IP country values were found "
        "in the historical dataset."
    )

    st.stop()


if not new_device_values:

    new_device_values = [
        "No",
        "Yes"
    ]


if not location_mismatch_values:

    location_mismatch_values = [
        "No",
        "Yes"
    ]


# ============================================================
# TRANSACTION INPUT COLUMNS
# ============================================================

transaction_col1, transaction_col2, transaction_col3 = (
    st.columns(3)
)


# ============================================================
# TRANSACTION COLUMN 1
# ============================================================

with transaction_col1:

    transaction_timestamp = st.date_input(
        "Transaction Timestamp",
        value=datetime.now()
    )

    source_currency = st.selectbox(
        "Source Currency",
        options=source_currency_values
    )

    dest_currency = st.selectbox(
        "Destination Currency",
        options=dest_currency_values
    )

    channel = st.selectbox(
        "Transaction Channel",
        options=channel_values
    )


# ============================================================
# TRANSACTION COLUMN 2
# ============================================================

with transaction_col2:

    amount_src = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.00,
        step=10.00,
        format="%.2f"
    )

    fee = st.number_input(
        "Transaction Fee",
        min_value=0.0,
        value=0.00,
        step=0.01,
        format="%.2f"
    )

    new_device = st.selectbox(
        "New Device",
        options=new_device_values
    )

    ip_country = st.selectbox(
        "IP Country",
        options=ip_country_values
    )


# ============================================================
# TRANSACTION COLUMN 3
# ============================================================

with transaction_col3:

    location_mismatch = st.selectbox(
        "Location Mismatch",
        options=location_mismatch_values
    )

    ip_risk_score = st.number_input(
        "IP Risk Score",
        min_value=0.0,
        value=safe_float(
            customer_record[
                "ip_risk_score"
            ]
        ),
        step=0.01,
        format="%.4f"
    )

    st.write("")

    st.write("")

    analyse_transaction = st.button(
        "🔍 Analyse Transaction",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PREPARE AND SEND API REQUEST
# ============================================================

if analyse_transaction:

    # ========================================================
    # VALIDATE CUSTOMER ACCOUNT INFORMATION
    # ========================================================

    home_country = customer_record[
        "home_country"
    ]

    kyc_tier = customer_record[
        "kyc_tier"
    ]

    if pd.isna(home_country):

        st.error(
            "The selected customer's Home Country "
            "is missing from the historical dataset."
        )

        st.stop()


    if pd.isna(kyc_tier):

        st.error(
            "The selected customer's KYC Tier "
            "is missing from the historical dataset."
        )

        st.stop()


    # ========================================================
    # CREATE PAYLOAD MATCHING PYDANTIC TransactionData
    # ========================================================

    payload = {
        "timestamp": transaction_timestamp.isoformat(),

        "customer_id": str(
            selected_customer
        ),

        "home_country": "ca",
                

        "source_currency": str(
            source_currency
        ),

        "dest_currency": str(
            dest_currency
        ),

        "channel": str(
            channel
        ),

        "amount_src": float(
            amount_src
        ),

        "fee": float(
            fee
        ),

        "new_device": str(
            new_device
        ),

        "ip_country": str(
            ip_country
        ),

        "location_mismatch": str(
            location_mismatch
        ),

        "ip_risk_score": float(
            ip_risk_score
        ),

        "kyc_tier": str(
            kyc_tier
        ),

        "account_age_days": safe_int(
            customer_record[
                "account_age_days"
            ]
        ),

        "device_trust_score": safe_float(
            customer_record[
                "device_trust_score"
            ]
        ),

        "risk_score_internal": safe_float(
            customer_record[
                "risk_score_internal"
            ]
        ),

        "corridor_risk": safe_float(
            customer_record[
                "corridor_risk"
            ]
        )
    }


    # ========================================================
    # DISPLAY API REQUEST
    # ========================================================

    with st.expander(
        "View API Request Payload"
    ):

        st.json(
            payload
        )


    # ========================================================
    # SEND REQUEST TO FASTAPI
    # ========================================================

    try:

        with st.spinner(
            "Analysing transaction with "
            "the fraud detection model..."
        ):

            response = requests.post(
                f"{API_URL.rstrip('/')}/predict",
                json=payload,
                timeout=60
            )


        # ====================================================
        # SUCCESSFUL RESPONSE
        # ====================================================

        if response.status_code == 200:

            result = response.json()


            # =================================================
            # EXTRACT API RESULTS
            # =================================================

            is_fraud = result.get(
                "is_fraud"
            )

            fraud_probability = result.get(
                "fraud_probability"
            )

            amount_usd = result.get(
                "amount_usd"
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


            # =================================================
            # FRAUD DETECTION RESULT
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '5. Fraud Detection Result'
                '</div>',
                unsafe_allow_html=True
            )


            if bool(is_fraud):

                st.error(
                    "🚨 FRAUDULENT TRANSACTION DETECTED"
                )

            else:

                st.success(
                    "✅ TRANSACTION APPEARS LEGITIMATE"
                )


            # =================================================
            # RESULT METRICS
            # =================================================

            result_col1, result_col2, result_col3, result_col4 = (
                st.columns(4)
            )


            # =================================================
            # PREDICTION
            # =================================================

            with result_col1:

                if bool(is_fraud):

                    st.metric(
                        "Prediction",
                        "FRAUD"
                    )

                else:

                    st.metric(
                        "Prediction",
                        "LEGITIMATE"
                    )


            # =================================================
            # FRAUD PROBABILITY
            # =================================================

            with result_col2:

                if fraud_probability is not None:

                    probability_percentage = (
                        float(
                            fraud_probability
                        )
                        * 100
                    )

                    st.metric(
                        "Fraud Probability",
                        f"{probability_percentage:.2f}%"
                    )

                else:

                    st.metric(
                        "Fraud Probability",
                        "N/A"
                    )


            # =================================================
            # AMOUNT USD
            # =================================================

            with result_col3:

                if amount_usd is not None:

                    st.metric(
                        "Amount (USD)",
                        f"${float(amount_usd):,.2f}"
                    )

                else:

                    st.metric(
                        "Amount (USD)",
                        "N/A"
                    )


            # =================================================
            # VELOCITY SPIKE
            # =================================================

            with result_col4:

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


            # =================================================
            # TRANSACTION VELOCITY
            # =================================================

            st.subheader(
                "Transaction Velocity Analysis"
            )


            velocity_col1, velocity_col2 = (
                st.columns(2)
            )


            with velocity_col1:

                if txn_velocity_1h is not None:

                    st.metric(
                        "Transactions in Last 1 Hour",
                        str(txn_velocity_1h)
                    )

                else:

                    st.metric(
                        "Transactions in Last 1 Hour",
                        "N/A"
                    )


            with velocity_col2:

                if txn_velocity_24h is not None:

                    st.metric(
                        "Transactions in Last 24 Hours",
                        str(txn_velocity_24h)
                    )

                else:

                    st.metric(
                        "Transactions in Last 24 Hours",
                        "N/A"
                    )


            # =================================================
            # COMPLETE API RESPONSE
            # =================================================

            with st.expander(
                "View Complete API Response"
            ):

                st.json(
                    result
                )


        # ====================================================
        # VALIDATION ERROR - HTTP 422
        # ====================================================

        elif response.status_code == 422:

            st.error(
                "FastAPI rejected the transaction "
                "because the submitted data does not "
                "match the TransactionData Pydantic model."
            )

            try:

                st.json(
                    response.json()
                )

            except ValueError:

                st.code(
                    response.text
                )


        # ====================================================
        # OTHER API ERRORS
        # ====================================================

        else:

            st.error(
                f"FastAPI returned HTTP "
                f"{response.status_code}."
            )

            try:

                st.json(
                    response.json()
                )

            except ValueError:

                st.code(
                    response.text
                )


    # ========================================================
    # CONNECTION ERROR
    # ========================================================

    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the FastAPI server.\n\n"
            "Please make sure FastAPI is running at:\n"
            f"{api_url}"
        )


    # ========================================================
    # TIMEOUT ERROR
    # ========================================================

    except requests.exceptions.Timeout:

        st.error(
            "The request to FastAPI timed out. "
            "Please check that the API and machine "
            "learning model are running correctly."
        )


    # ========================================================
    # REQUEST ERROR
    # ========================================================

    except requests.exceptions.RequestException as e:

        st.error(
            "An error occurred while communicating "
            f"with FastAPI: {e}"
        )


    # ========================================================
    # UNEXPECTED ERROR
    # ========================================================

    except Exception as e:

        st.error(
            f"An unexpected error occurred: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Finlora Fraud Detection System | "
    "Streamlit Frontend + FastAPI Machine Learning API")