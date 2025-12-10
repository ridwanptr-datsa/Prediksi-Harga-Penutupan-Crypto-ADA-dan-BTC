import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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

    open_val = st.number_input("Open", value=0.0, format="%.12f")
    high_val = st.number_input("High", value=0.0, format="%.12f")
    low_val = st.number_input("Low", value=0.0, format="%.12f")
    ticker_val = st.selectbox("Ticker", ["BTC"])

    st.markdown('</div>', unsafe_allow_html=True)

    # Tombol Prediksi
    pred_btn = st.button("Prediksi", use_container_width=True)

# ================= KOLOM OUTPUT: GRAFIK + HASIL PREDIKSI (KANAN) =================
with col2:

    st.markdown("### 📈 Ilustrasi Grafik Fitur")

    # Jika tombol belum ditekan → tidak ada grafik
    if not pred_btn:
        st.info("Masukkan input dan tekan **Prediksi** untuk menampilkan ilustrasi grafik fitur.")
        st.markdown("### 🎯 Hasil Prediksi")
        st.info("Masukkan input dan tekan **Prediksi** untuk hasil prediksi harga penutupan (close).")
    else:
        # ================= MULAI VISUALISASI SETELAH PREDIKSI =================

        preview_df = pd.DataFrame([[0] * len(model_features)], columns=model_features)

        if "open" in preview_df.columns: preview_df["open"] = open_val
        if "high" in preview_df.columns: preview_df["high"] = high_val
        if "low" in preview_df.columns: preview_df["low"] = low_val

        ticker_col = f"ticker_{ticker_val}"
        if ticker_col in preview_df.columns:
            preview_df[ticker_col] = 1

        try:
            live_prediction = model.predict(preview_df)[0]
        except:
            live_prediction = 0

        x_points = ["Open", "High", "Low", "Predicted Close"]
        y_values = [open_val, high_val, low_val, live_prediction]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_points,
            y=y_values,
            mode='lines+markers',
            marker=dict(size=10),
            line=dict(width=3),
            name="Live Update"
        ))

        fig.update_layout(
            title="Perubahan Fitur & Prediksi",
            xaxis_title="Fitur",
            yaxis_title="Harga (USD)",
            template="plotly_white",
            hovermode="x"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ----------------- Hasil Prediksi -----------------
        st.markdown("### 🎯 Hasil Prediksi")
        st.markdown(
            f'''
            <div class="card" style="text-align:center;">
                <h2 style="color:#2E86C1;">Prediksi Close Price</h2>
                <h1 style="color:#4CAF50; font-size:42px;">{live_prediction:.12f}</h1>
            </div>
            ''',
            unsafe_allow_html=True
        )
