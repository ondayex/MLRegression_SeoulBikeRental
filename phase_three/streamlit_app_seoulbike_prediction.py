import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Load the saved model
@st.cache_resource
def load_model():
    with open('bike_rental_model_xgboost.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title('Bike Rental Prediction App')

st.write("""
This app predicts the number of bike rentals based on various features.
Please input the required information below.
""")

# Input features
col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Date", datetime.now())
    temperature = st.slider("Temperature (°C)", -20.0, 40.0, 20.0)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0)
    visibility = st.slider("Visibility (10m)", 0, 2000, 1000)

with col2:
    solar_radiation = st.slider("Solar Radiation (MJ/m2)", 0.0, 5.0, 1.0)
    rainfall = st.slider("Rainfall (mm)", 0.0, 100.0, 0.0)
    snowfall = st.slider("Snowfall (cm)", 0.0, 30.0, 0.0)
    seasons = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
    holiday = st.selectbox("Holiday", ["No Holiday", "Holiday"])

# Prepare input data
input_data = pd.DataFrame({
    'Date': [date.strftime("%A")],
    'Temperature(°C)': [temperature],
    'Humidity(%)': [humidity],
    'Wind speed (m/s)': [wind_speed],
    'Visibility (10m)': [visibility],
    'Solar Radiation (MJ/m2)': [solar_radiation],
    'Rainfall(mm)': [rainfall],
    'Snowfall (cm)': [snowfall],
    'Seasons': [seasons],
    'Holiday': [holiday]
})

# Make prediction
if st.button('Predict Bike Rentals'):
    prediction = model.predict(input_data)
    st.success(f"Predicted number of bike rentals: {int(prediction[0])}")

st.write("""
### Note:
This prediction is based on historical data and may not account for current events or changes in bike rental patterns.
""")