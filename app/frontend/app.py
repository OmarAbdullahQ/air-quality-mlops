import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Riyadh Air Quality",
    page_icon="🌫️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    h1 {
        text-align: center;
    }

    div[data-testid="stForm"] {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 20px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌫️ Riyadh Air Quality")
st.caption("Predict whether air pollution will be high during the next hour.")

with st.form("prediction_form"):

    left, right = st.columns(2)

    with left:
        pm2_5 = st.number_input("PM2.5", value=25.0)
        pm10 = st.number_input("PM10", value=60.0)
        temperature = st.number_input("Temperature (°C)", value=32.0)
        humidity = st.number_input("Humidity (%)", value=25.0)
        wind = st.number_input("Wind Speed", value=12.0)

    with right:
        lag_1 = st.number_input("PM2.5 (1 Hour Ago)", value=24.0)
        lag_3 = st.number_input("PM2.5 (3 Hours Ago)", value=22.0)
        rolling = st.number_input("6-Hour Average", value=23.0)
        hour = st.slider("Hour", 0, 23, 12)
        day = st.slider("Day of Week", 0, 6, 2)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:

    payload = {
        "pm2_5": pm2_5,
        "pm10": pm10,
        "temperature_2m": temperature,
        "relative_humidity_2m": humidity,
        "wind_speed_10m": wind,
        "hour": hour,
        "day_of_week": day,
        "pm2_5_lag_1": lag_1,
        "pm2_5_lag_3": lag_3,
        "pm2_5_rolling_mean_6": rolling,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )

        response.raise_for_status()
        result = response.json()

        st.divider()
        st.subheader("Prediction")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Probability",
                f"{result['probability']:.1%}",
            )

        with c2:
            st.metric(
                "Risk Level",
                result["risk_level"].upper(),
            )

        if result["prediction"]:
            st.error("⚠️ High pollution expected.")
        else:
            st.success("✅ Air quality expected to remain normal.")

    except requests.RequestException as error:
        st.error(f"API request failed:\n{error}")

st.divider()

try:
    health = requests.get(
        f"{API_URL}/health",
        timeout=5,
    ).json()

    if health["status"] == "ok":
        st.success("🟢 API Online")
    else:
        st.error("🔴 API Offline")

except requests.RequestException:
    st.error("🔴 API Unavailable")