 
import streamlit as st 
import pandas as pd 
import pickle

# Load trained Linear Regression model 
model = pickle.load(open("linear_regression_model.pkl", "rb"))

# Streamlit app title 
st.title("Aplikasi Prediksi Harga Penutupan Crypto (Linear Regression)")

# Sidebar for user inputs 
st.sidebar.header("Input Fitur")
open_val = st.sidebar.number_input("Open", value=0.0)
high_val = st.sidebar.number_input("High", value=0.0)
low_val = st.sidebar.number_input("Low", value=0.0)

ticker_val = st.sidebar.selectbox("Ticker", ["ADA", "BTC"])

# Define the exact columns and their dtypes expected by the model during training 
# This list ensures correct order and includes all dummy variables used during training 
columns = [
    "open", "high", "low", "volume",
    "ticker_ADA", "ticker_BTC"
]

# Create an empty DataFrame with the correct columns and dtypes 
input_df = pd.DataFrame({col: [0] for col in columns})

# Add a single row of data, initially all zeros 
# (already done above by initializing with zeros)

# Populate numerical features 
input_df["open"] = open_val
input_df["high"] = high_val
input_df["low"] = low_val

# Populate one-hot encoded categorical features 
input_df[f"ticker_{ticker_val}"] = 1

# Make prediction 
prediction = model.predict(input_df)[0]

st.subheader("Hasil Prediksi Harga Penutupan Crypto")
st.write(prediction)
