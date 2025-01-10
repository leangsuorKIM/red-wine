import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Input values
fixed_acidity = st.number_input(label="fixed acidity", value=7.2)
volatile_acidity = st.number_input(label="volatile acidity", value=0.660)
citric_acid = st.number_input(label="citric acid", value=0.03)
residual_sugar = st.number_input(label="residual sugar", value=2.3)
chlorides = st.number_input(label="chlorides", value=0.078)
free_sulfur_dioxide = st.number_input(label="free sulfur dioxide", value=16.0)
total_sulfur_dioxide = st.number_input(label="total sulfur dioxide", value=86.0)
density = st.number_input(label="density", value=0.99743)
pH = st.number_input(label="pH", value=3.53)
sulphates = st.number_input(label="sulphates", value=0.57)
alcohol = st.number_input(label="alcohol", value=9.7)

# Create the numerical input array (X_num)
X_num = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                   chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                   pH, sulphates, alcohol]], dtype=np.float32)

# Load scaler and apply transformation
with open(file="ss.pkl", mode="rb") as ss_file:
    ss = pickle.load(file=ss_file)
X_scaled = ss.transform(X_num)

# Load label encoder if needed (for categorical encoding)
with open(file="le.pkl", mode="rb") as le_file:
    le = pickle.load(file=le_file)

# Load the pre-trained Linear Regression model
with open(file="lr.pkl", mode="rb") as lr_file:
    lr = pickle.load(file=lr_file)

# Predict the output
y_pred = lr.predict(X_scaled)

# If needed, you can transform the result back (e.g., if it's log-scaled, for instance)
y_raw = 1 / y_pred  # Assuming this is the intended transformation

# Clamp the result to be between 2 and 8 (since the quality should be in this range)
y_raw_clamped = np.clip(np.round(y_raw), 2, 8).astype(int)  # Round and ensure it's between 2 and 8

# Combine the scaled features and prediction into a DataFrame
data = np.concatenate([X_scaled, y_raw_clamped.reshape(-1, 1)], axis=1)
df = pd.DataFrame(
    data=data,
    columns=[
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'quality_pred'
    ]
)

# Display the result
st.write(df)
