import streamlit as st
import numpy as np
import pandas as pd
import pickle

fixed_acidity  = st.number_input(label="fixed acidity", value=7.2)
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
X_num = np.array(
    object=[
        [
            'fixed acidity',
            'volatile acidity',
            'citric acid',
            'residual sugar',
            'chlorides',
            'free sulfur dioxide',
            'total sulfur dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol'
        ]
    ],
    dtype=np.float32,
)
with open(file="ss.pkl", mode="rb") as ss_file:
    ss = pickle.load(file=ss_file)
X1 = ss.transform(X_num)
with open(file="le.pkl", mode="rb") as le_file:
    le = pickle.load(file=le_file)
X_cat = np.array(object=[origin])
X2 = le.transform(X_cat)
# st.write(X_cat)
X = np.concat([X1, X2.reshape(-1, 1)], axis=1)
# st.write(X)
with open(file="lr.pkl", mode="rb") as lr_file:
    lr = pickle.load(file=lr_file)
y = lr.predict(X)
X_raw = np.concat([X_num, X_cat.reshape(-1, 1)], axis=1)
y_raw = 1 / y
data = np.concat([X_raw, y_raw.reshape(-1, 1)], axis=1)
df = pd.DataFrame(
    data=data,
    columns=[
        'fixed acidity' ,
        'volatile acidity' ,
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide' ,
        'total sulfur dioxide' ,
        'density' ,
        'pH' ,
        'sulphates' ,
        'alcohol' ,
        "quality_pred",
    ],
)
st.write(df)
