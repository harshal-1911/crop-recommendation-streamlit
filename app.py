import streamlit as st
import numpy as np
import pickle

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Page config
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾")

st.title("🌾 Crop Recommendation System")
st.write("Enter soil and environmental details to get the best crop recommendation.")

# User inputs
N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0)
rainfall = st.number_input("Rainfall (mm)")

# Prediction
if st.button("🌱 Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    crop = le.inverse_transform(prediction)

    st.success(f"✅ Recommended Crop: **{crop[0].upper()}**")
