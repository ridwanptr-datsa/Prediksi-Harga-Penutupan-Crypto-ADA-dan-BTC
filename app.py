import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load trained model
model = pickle.load(open("linear_regression_model.pkl", "rb"))

# Ambil fitur yang digunakan saat training
model_features = model.feature_names_in_

# === UI Styling ===
st.set_page_config(page_title="Prediksi Harga Crypto", layout="wide")

st.markdown(
    '''
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
        color: #555;
    }
    .card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# === Title ===
st.markdown('<div class="main-title">Aplikasi Prediksi Harga Penutupan Crypto (BTC)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Model Machine Learning — Linear Regression</div>', unsafe_allow_html=True)

# =====================================================================
#                  BAGI TAMPILAN MENJADI DUA KOLOM
# =====================================================================
col1, col2 = st.columns([1, 1])

# ======================== KOLOM INPUT (KIRI) ==========================
with col1:
    st.markdown("### 🔧 Input Fitur")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    open_val = st.number_input("Open", value=0.0, format="%.12f")
    high_val = st.number_input("High", value=0.0, format="%.12f")
    low_val = st.number_input("Low", value=0.0, format="%.12f")
    ticker_val = st.selectbox("Ticker", ["BTC"])

    st.markdown('</div>', unsafe_allow_html=True)

    # Tombol Prediksi
    pred_btn = st.button("Prediksi", use_container_width=True)

# ================= KOLOM OUTPUT: GRAFIK + HASIL PREDIKSI (KANAN) =================
with col2:

    # --- Judul Grafik tetap tampil ---
    st.markdown("### 📈 Ilustrasi Grafik Fitur")

    if pred_btn:

        # Buat DataFrame kosong sesuai fitur model
        input_df = pd.DataFrame([[0] * len(model_features)], columns=model_features)

        # Isi fitur numerik
        if "open" in input_df.columns:
            input_df["open"] = open_val
        if "high" in input_df.columns:
            input_df["high"] = high_val
        if "low" in input_df.columns:
            input_df["low"] = low_val

        # One-hot encode ticker
        ticker_col = f"ticker_{ticker_val}"
        if ticker_col in input_df.columns:
            input_df[ticker_col] = 1

        # Lakukan prediksi
        prediction = model.predict(input_df)[0]

        # === GRAFIK ===
        fig, ax = plt.subplots(figsize=(8, 4))
        x_points = ["Open", "High", "Low", "Predicted Close"]
        y_values = [open_val, high_val, low_val, prediction]

        ax.plot(x_points, y_values, marker='o', linewidth=2)
        ax.set_title("Ilustrasi Nilai Open, High, Low, dan Prediksi Close", fontsize=12)
        ax.set_ylabel("Harga (USD)")
        ax.grid(True, linestyle="--", alpha=0.3)

        st.pyplot(fig)

    else:
        # Placeholder grafik sebelum prediksi
        st.info("Masukkan input dan tekan **Prediksi** untuk menampilkan grafik.")

    # --- Judul Hasil Prediksi tetap tampil ---
    st.markdown("### 🎯 Hasil Prediksi Close")

    if pred_btn:
        st.markdown(
            f'''
            <div class="card" style="text-align:center;">
                <h2 style="color:#2E86C1;">Prediksi Close Price</h2>
                <h1 style="color:#4CAF50; font-size:42px;">{prediction:.12f}</h1>
            </div>
            ''',
            unsafe_allow_html=True
        )
    else:
        st.warning("Belum ada prediksi. Isi data lalu tekan tombol **Prediksi**.")
