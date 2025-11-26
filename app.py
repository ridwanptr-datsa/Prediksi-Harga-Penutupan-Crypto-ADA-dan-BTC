
import streamlit as st
import pandas as pd
import pickle

# Load trained Random Forest model
model = pickle.load(open("gradient_boosting_regressor_model.pkl", "rb"))

# Streamlit app title
st.title("Aplikasi Prediksi Harga Penutupan Crypto")

st.write("Masukkan data untuk memprediksi nilai **close price**.")

# Sidebar inputs
st.sidebar.header("Input Fitur")

open_val = st.sidebar.number_input("Open", value=0.0)
high_val = st.sidebar.number_input("High", value=0.0)
low_val = st.sidebar.number_input("Low", value=0.0)

ticker_val = st.sidebar.selectbox("Ticker", ["ADA", "BTC"])

# Define columns used during model training
columns = [
    "open", "high", "low", "volume",
    "daily_return", "range",
    "ticker_ADA", "ticker_BTC"
]

# Initialize empty DataFrame with correct columns
input_df = pd.DataFrame({col: [0] for col in columns})

# Populate numeric features
input_df["open"] = open_val
input_df["high"] = high_val
input_df["low"] = low_val

# Populate categorical one-hot columns
input_df[f"ticker_{ticker_val}"] = 1

# Prediction
prediction = model.predict(input_df)[0]

st.subheader("Hasil Prediksi Harga Penutupan Crypto ADA dan BTC")
st.write(prediction)

