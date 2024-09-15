import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Load the trained pipeline model
with open('E:/CP05_seasonal_bike_rental/phase_three/bike_rental_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a simple input form for the user to provide feature values
st.title("Seoul Bike Rental Prediction App")

# Input form for user data
hour = st.slider('Hour', 0, 23, 12)
temperature = st.number_input('Temperature (°C)', min_value=-30.0, max_value=40.0)
humidity = st.number_input('Humidity (%)', min_value=0, max_value=100)
wind_speed = st.number_input('Wind Speed (m/s)', min_value=0.0, max_value=20.0)
visibility = st.number_input('Visibility (10m)', min_value=0, max_value=2000)
solar_radiation = st.number_input('Solar Radiation (MJ/m2)', min_value=0.0, max_value=3.0)
rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=100.0)
snowfall = st.number_input('Snowfall (cm)', min_value=0.0, max_value=50.0)

# Dropdown options for categorical inputs
season = st.selectbox('Season', ['Spring', 'Summer', 'Autumn', 'Winter'])
holiday = st.selectbox('Holiday', ['Holiday', 'No Holiday'])

# Date input to automatically detect the day of the week
date_input = st.date_input('Select a date', datetime.today())
day_of_week = date_input.weekday()  # Monday=0, Sunday=6

# Create a DataFrame with the user input (including day of the week)
input_data = pd.DataFrame([[hour, temperature, humidity, wind_speed, visibility,
                            solar_radiation, rainfall, snowfall, season, holiday, day_of_week]],
                          columns=['Hour', 'Temperature', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
                                   'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 
                                   'Seasons', 'Holiday', 'Date'])

# Make a prediction using the trained model
prediction = model.predict(input_data)

# Display the prediction
st.write(f"Predicted Number of Bike Rentals: {int(prediction[0]):,}")
