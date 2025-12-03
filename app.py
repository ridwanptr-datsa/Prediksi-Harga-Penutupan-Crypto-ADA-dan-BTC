import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("linear_regression_model.pkl", "rb"))

# Ambil fitur yang digunakan saat training
model_features = model.feature_names_in_

# Streamlit title
st.title("Aplikasi Prediksi Harga Penutupan Crypto (Linear Regression)")

# Sidebar input
st.sidebar.header("Input Fitur")

open_val = st.sidebar.number_input("Open", value=0.0)
high_val = st.sidebar.number_input("High", value=0.0)
low_val = st.sidebar.number_input("Low", value=0.0)
close_val = st.sidebar.number_input("Close", value=0.0)

ticker_val = st.sidebar.selectbox("Ticker", ["ADA", "BTC"])

# Buat DataFrame kosong sesuai fitur model
input_df = pd.DataFrame([[0] * len(model_features)], columns=model_features)

# Isi fitur numerik jika ada di model
if "open" in input_df.columns:
    input_df["open"] = open_val

if "high" in input_df.columns:
    input_df["high"] = high_val

if "low" in input_df.columns:
    input_df["low"] = low_val

if "close" in input_df.columns:
    input_df["close"] = close_val

# One-hot encoding ticker jika kolomnya ada
ticker_col = f"ticker_{ticker_val}"
if ticker_col in input_df.columns:
    input_df[ticker_col] = 1

# Prediksi
prediction = model.predict(input_df)[0]

st.subheader("Hasil Prediksi Harga Penutupan Crypto")
st.write(prediction)
