
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
close_val = st.sidebar.number_input("Close", value=0.0)

ticker_val = st.sidebar.selectbox("Ticker", ["ADA", "BTC"])

# Define model feature columns (same order used during training)
columns = [
    "open", "high", "low", "close",
    "ticker_ADA", "ticker_BTC"
]

# Create DataFrame with initial zero values
input_df = pd.DataFrame({col: [0] for col in columns})

# Fill numeric features
input_df["open"] = open_val
input_df["high"] = high_val
input_df["low"] = low_val
input_df["close"] = close_val

# One-hot encode ticker
input_df[f"ticker_{ticker_val}"] = 1

# Make prediction
prediction = model.predict(input_df)[0]

st.subheader("Hasil Prediksi (Close Price Crypto)")
st.write(prediction)
