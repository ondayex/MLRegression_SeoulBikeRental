import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import pickle

# Load dataset
dataset = pd.read_csv('E:/CP05_seasonal_bike_rental/dataset/SeoulBikeData.csv')

# Drop the 'Rented Bike Count', 'Functioning Day', and 'Dew point temperature' columns
X = dataset.drop(['Rented Bike Count', 'Functioning Day', 'Dew point temperature'], axis=1)
y = dataset['Rented Bike Count']

# Convert 'Date' column to datetime if it's not already
X['Date'] = pd.to_datetime(dataset['Date'], format="%d/%m/%Y").dt.day_name()

# Define categorical columns for One-Hot Encoding
categorical_cols = ['Seasons', 'Holiday', 'Date']

# Preprocessing pipeline for categorical columns (dropping first category to avoid multicollinearity)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)  # drop='first' will drop Monday (0)
    ], remainder='passthrough')

# Define the XGBoost regression model (using default parameters)
model = XGBRegressor()

# Create a pipeline with the preprocessor and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Save the pipeline (preprocessing + model) as a pickle file
with open('bike_rental_model_xgboost.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
