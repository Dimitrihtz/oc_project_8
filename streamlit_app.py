import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
THRESHOLD = 0.10

FEATURE_IMPORTANCE = {
    "EXT_SOURCES_MEAN": 25.48,
    "CREDIT_TERM": 2.84,
    "EXT_SOURCE_3": 2.58,
    "GOODS_PRICE_CREDIT_PERCENT": 1.59,
    "INSTAL_AMT_PAYMENT_sum": 1.25,
    "AMT_ANNUITY": 1.24,
    "POS_CNT_INSTALMENT_FUTURE_mean": 1.24,
    "DAYS_BIRTH": 1.24,
    "EXT_SOURCES_WEIGHTED": 1.18,
    "EXT_SOURCE_2": 1.17,
}


@st.cache_data(ttl=30)
def check_health():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def create_gauge(probability):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".2f"},
            title={"text": "Default Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 10], "color": "#2ecc71"},
                    {"range": [10, 30], "color": "#f39c12"},
                    {"range": [30, 100], "color": "#e74c3c"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": THRESHOLD * 100,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=60, b=20, l=30, r=30))
    return fig


def create_feature_importance_chart():
    features = list(FEATURE_IMPORTANCE.keys())[::-1]
    values = list(FEATURE_IMPORTANCE.values())[::-1]
    fig = go.Figure(
        go.Bar(x=values, y=features, orientation="h", marker_color="#3498db")
    )
    fig.update_layout(
        title="Top 10 Feature Importance (%)",
        xaxis_title="Importance (%)",
        height=400,
        margin=dict(t=40, b=40, l=10, r=10),
    )
    return fig


# --- Sidebar ---
with st.sidebar:
    st.header("API Status")
    health = check_health()
    if health and health.get("model_loaded"):
        st.success("API Connected")
    else:
        st.error("API Unavailable")

    st.plotly_chart(create_feature_importance_chart(), use_container_width=True)

# --- Main ---
st.title("Credit Scoring Dashboard")
st.markdown(
    "Enter the applicant's features below and submit to get a credit decision."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        ext_sources_mean = st.slider(
            "EXT_SOURCES_MEAN",
            min_value=0.0, max_value=1.0, value=0.524, step=0.001,
            help="Mean of external credit scores",
        )
        credit_term = st.slider(
            "CREDIT_TERM",
            min_value=0.0, max_value=1.0, value=0.05, step=0.001,
            help="Credit term ratio (annuity / credit amount)",
        )
        ext_source_3 = st.slider(
            "EXT_SOURCE_3",
            min_value=0.0, max_value=1.0, value=0.535, step=0.001,
            help="External source 3 score",
        )
        goods_price_credit_pct = st.slider(
            "GOODS_PRICE_CREDIT_PERCENT",
            min_value=0.0, max_value=1.5, value=0.9, step=0.01,
            help="Goods price as percentage of credit amount",
        )
        ext_sources_weighted = st.slider(
            "EXT_SOURCES_WEIGHTED",
            min_value=0.0, max_value=3.0, value=1.5, step=0.01,
            help="Weighted combination of external sources",
        )

    with col2:
        instal_amt_payment_sum = st.number_input(
            "INSTAL_AMT_PAYMENT_sum",
            min_value=0.0, max_value=1e8, value=318619.5, step=1000.0,
            help="Sum of installment payments",
        )
        amt_annuity = st.number_input(
            "AMT_ANNUITY",
            min_value=0.01, max_value=1e6, value=24903.0, step=100.0,
            help="Loan annuity amount",
        )
        days_birth = st.number_input(
            "DAYS_BIRTH",
            min_value=-30000, max_value=-1, value=-15750, step=1,
            help="Client age in days (negative, relative to application date)",
        )
        st.caption(f"Approx. age: {abs(days_birth) / 365.25:.1f} years")
        pos_cnt_instalment_future = st.slider(
            "POS_CNT_INSTALMENT_FUTURE_mean",
            min_value=0.0, max_value=200.0, value=6.95, step=0.05,
            help="Mean count of future installments (POS)",
        )
        ext_source_2 = st.slider(
            "EXT_SOURCE_2",
            min_value=0.0, max_value=1.0, value=0.566, step=0.001,
            help="External source 2 score",
        )

    submitted = st.form_submit_button("Get Prediction", use_container_width=True)

if submitted:
    payload = {
        "EXT_SOURCES_MEAN": ext_sources_mean,
        "CREDIT_TERM": credit_term,
        "EXT_SOURCE_3": ext_source_3,
        "GOODS_PRICE_CREDIT_PERCENT": goods_price_credit_pct,
        "INSTAL_AMT_PAYMENT_sum": instal_amt_payment_sum,
        "AMT_ANNUITY": amt_annuity,
        "POS_CNT_INSTALMENT_FUTURE_mean": pos_cnt_instalment_future,
        "DAYS_BIRTH": days_birth,
        "EXT_SOURCES_WEIGHTED": ext_sources_weighted,
        "EXT_SOURCE_2": ext_source_2,
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

        if resp.status_code == 422:
            st.error("Validation error: check that all values are within range.")
            st.json(resp.json())
        else:
            resp.raise_for_status()
            data = resp.json()

            probability = data["probability_default"]
            prediction = data["prediction"]
            decision = data["credit_decision"]

            if decision == "approved":
                st.success(f"Credit Decision: **APPROVED**")
            else:
                st.error(f"Credit Decision: **DENIED**")

            m1, m2 = st.columns(2)
            m1.metric("Default Probability", f"{probability:.4%}")
            m2.metric("Prediction Class", prediction)

            st.plotly_chart(create_gauge(probability), use_container_width=True)

    except requests.ConnectionError:
        st.error(
            "Cannot connect to the API. "
            "Make sure it is running at " + API_URL
        )
    except requests.Timeout:
        st.error("Request timed out. The API may be overloaded.")
    except requests.HTTPError as e:
        st.error(f"API error: {e}")
