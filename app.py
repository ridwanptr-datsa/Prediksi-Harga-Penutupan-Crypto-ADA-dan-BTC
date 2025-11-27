import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# ===============================
# Load trained model
# ===============================
model = pickle.load(open("gradient_boosting_regressor_model.pkl", "rb"))

# ===============================
# Streamlit App UI
# ===============================
st.title("Aplikasi Prediksi Harga Penutupan Crypto (ADA & BTC)")

st.write("Masukkan fitur untuk memprediksi **close price**.")

# Sidebar input
st.sidebar.header("Input Fitur")

date_val = st.sidebar.date_input("Date")
open_val = st.sidebar.number_input("Open", value=0.0)
high_val = st.sidebar.number_input("High", value=0.0)
low_val = st.sidebar.number_input("Low", value=0.0)
volume_val = st.sidebar.number_input("Volume", value=0.0)
daily_return_val = st.sidebar.number_input("Daily Return", value=0.0)
range_val = st.sidebar.number_input("Range", value=0.0)

ticker_val = st.sidebar.selectbox("Ticker", ["ADA", "BTC"])

# ===============================
# Build all columns used during training
# ===============================

# Tahun yang ada pada dataset (2017 sampai 2025)
year_cols = [f"year_{y}" for y in range(2017, 2026)]

# Bulan 1 sampai 12
month_cols = [f"month_{m}" for m in range(1, 13)]

# Dummy ticker
ticker_cols = ["ticker_ADA", "ticker_BTC"]

# Semua fitur model
all_columns = (
    ["open", "high", "low", "volume", "daily_return", "range"] +
    ticker_cols + year_cols + month_cols
)

# Buat DataFrame satu baris dengan nilai 0
input_df = pd.DataFrame({col: [0] for col in all_columns})

# ===============================
# Isi nilai numerik
# ===============================
input_df["open"] = open_val
input_df["high"] = high_val
input_df["low"] = low_val
input_df["volume"] = volume_val
input_df["daily_return"] = daily_return_val
input_df["range"] = range_val

# ===============================
# Categorical: ticker
# ===============================
input_df[f"ticker_{ticker_val}"] = 1

# ===============================
# Extract year & month from date
# ===============================
year = date_val.year
month = date_val.month

if f"year_{year}" in input_df.columns:
    input_df[f"year_{year}"] = 1

if f"month_{month}" in input_df.columns:
    input_df[f"month_{month}"] = 1

# ===============================
# Prediction
# ===============================
prediction = model.predict(input_df)[0]

st.subheader("Hasil Prediksi Close Price")
st.write(prediction)
