import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import os

st.title('Bike Rental Prediction App')

# Check if the model file exists
model_path = '/phase_three/bike_rental_model_xgboost.pkl'
model_exists = os.path.exists(model_path)

if not model_exists:
    st.error(f"Model file '{model_path}' not found. Please ensure the model file is in the same directory as this script.")
    st.stop()

# Load the saved model
@st.cache_resource
def load_model():
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

st.write("""
This app predicts the number of bike rentals based on various features.
Please input the required information below.
""")

# Input features
col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Date", datetime.now())
    hour = st.slider("Hour", 0, 23, 12)
    temperature = st.slider("Temperature (°C)", -20.0, 40.0, 20.0)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    wind_speed = st.slider("Wind speed (m/s)", 0.0, 10.0, 2.0)
    visibility = st.slider("Visibility (10m)", 0, 2000, 1000)

with col2:
    solar_radiation = st.slider("Solar Radiation (MJ/m2)", 0.0, 5.0, 1.0)
    rainfall = st.slider("Rainfall (mm)", 0.0, 100.0, 0.0)
    snowfall = st.slider("Snowfall (cm)", 0.0, 30.0, 0.0)
    seasons = st.selectbox("Seasons", ["Spring", "Summer", "Autumn", "Winter"])
    holiday = st.selectbox("Holiday", ["No Holiday", "Holiday"])

# Prepare input data
input_data = pd.DataFrame({
    'Date': [date.strftime("%A")],
    'Hour': [hour],
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
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted number of bike rentals: {int(prediction[0])}")
    except Exception as e:
        st.error(f"An error occurred while making the prediction: {str(e)}")
        st.write("Debug information:")
        st.write(f"Input data columns: {input_data.columns.tolist()}")
        st.write("Please ensure that the input features match those used during model training.")

st.write("""
### Note:
This prediction is based on historical data and may not account for current events or changes in bike rental patterns.
""")

st.write("""
### How to add the model file:
If you're seeing an error about a missing model file, follow these steps:
1. Ensure you have run the 'export_xgboost_model.py' script to generate the 'bike_rental_model_xgboost.pkl' file.
2. Add the 'bike_rental_model_xgboost.pkl' file to your GitHub repository in the same directory as this Streamlit app.
3. Commit and push the changes to GitHub.
4. Redeploy your Streamlit app or wait for automatic deployment (if set up).
""")