import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import time

# Set page config
st.set_page_config(page_title="🔆 Solar Power Forecast", layout="centered")

# Load the trained model and scaler
model = joblib.load('solar_Power_eneration_Forecasting_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title & Description
st.title("🔆 Solar Power Generation Forecast")
st.markdown("This app predicts DC power output based on solar and environmental conditions using a trained machine learning model.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("Adjust your input parameters below.")
    st.markdown("---")
    st.markdown("💡 The model currently uses a Random Forest Regressor.")
    st.info("Format date as YYYY-MM-DD HH:MM")

# Input Section
st.markdown("## 🧾 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    IRRADIATION = st.number_input("☀️ Irradiation (W/m²)", value=500.0)
    MODULE_TEMPERATURE = st.number_input("🌡️ Module Temperature (°C)", value=25.0)

with col2:
    AMBIENT_TEMPERATURE = st.number_input("🌤️ Ambient Temperature (°C)", value=20.0)
    date_str = st.text_input("📅 Date & Time (YYYY-MM-DD HH:MM)", value="2025-07-18 14:00")

# Prediction button
if st.button("🔍 Predict Power Output"):
    try:
        # Parse date
        input_time = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        hour = input_time.hour
        day = input_time.day
        month = input_time.month
        day_of_week = input_time.weekday()

        # Build input dataframe
        input_df = pd.DataFrame([[IRRADIATION, MODULE_TEMPERATURE, AMBIENT_TEMPERATURE, hour, day, month, day_of_week]],
            columns=['IRRADIATION','MODULE_TEMPERATURE','AMBIENT_TEMPERATURE','HOUR','DAY','MONTH','DAY_OF_WEEK'])

        # Scale and predict
        with st.spinner("Running prediction..."):
            time.sleep(1.2)  # Just for visual effect
            input_scaled = scaler.transform(input_df)
            predicted_dc_power = model.predict(input_scaled)

        # Display result
        st.success("✅ Prediction complete!")
        st.metric(label="🔋 Predicted DC Power (kW)", value=f"{predicted_dc_power[0]:.2f}")

        

    except ValueError:
        st.error("❌ Invalid date format. Please use YYYY-MM-DD HH:MM")

