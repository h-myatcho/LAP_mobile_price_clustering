import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Load your trained model for mobile price clustering
mobile_price_model = joblib.load("mobile_model.pkl")

# Set up the Streamlit app
st.title('Mobile Price Clustering')

# Get user inputs for the features
battery_power = st.slider('Battery Power (mAh)', 501, 1998)
fc = st.slider('Front Camera (Mega Pixels)', 0, 19)
int_memory = st.slider('Internal Memory (GB)', 2, 64)
m_dep = st.slider('Mobile Depth (cm)', 0.1, 1.0)
mobile_wt = st.slider('Mobile Weight (g)', 80, 200)
pc = st.slider('Primary Camera (Mega Pixels)', 0, 20)
ram = st.slider('RAM (Mega Bytes)', 256, 3998)
sc_h = st.slider('Screen Height (cm)', 5, 19)
sc_w = st.slider('Screen Width (cm)', 0, 18)
talk_time = st.slider('Talk Time (hours)', 2, 10)
dual_sim = st.radio('Dual SIM Support', ['Yes', 'No'])
four_g = st.radio('4G Support', ['Yes', 'No'])
wifi = st.radio('Wi-Fi Support', ['Yes', 'No'])
price = st.radio('Price Range', ['Very high cost', 'High cost', 'Medium cost', 'Low cost'])

# Preprocess the user inputs
dual_sim = 1 if dual_sim == 'Yes' else 0
four_g = 1 if four_g == 'Yes' else 0
wifi = 1 if wifi == 'Yes' else 0
if price == 'Very high cost':
    price = 3 
elif price == 'High cost':
    price = 2
elif price == 'Medium cost':
    price = 1
else:
    price = 0

# Add a button for prediction
button_clicked = st.button("Predict")

# Perform prediction when the button is clicked
if button_clicked:
    # Create a feature vector
    data = [
        battery_power, dual_sim, fc, four_g, int_memory,
        m_dep, mobile_wt, pc, ram, sc_h, sc_w, talk_time, wifi, price
    ]

    # Predict the mobile price cluster
    cluster_assignment = mobile_price_model.predict([data])

    # Define the cluster labels
    cluster_labels = {
        0: 'Mobile Group 1',
        1: 'Mobile Group 2',
        2: 'Mobile Group 3'
    }

    # Display the predicted cluster
    st.success(f'Based on the provided features, your mobile price cluster is: {cluster_labels[cluster_assignment[0]]}')
