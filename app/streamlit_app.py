import streamlit as st
import numpy as np
import datetime
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objs as go

from utils.fetch_data import fetch_data
from utils.preprocess import preprocess_data

st.title("💰 Kripto Para Tahmin Uygulaması")

coin = st.selectbox("Kripto para seçin:", ["bitcoin", "ethereum", "solana"])

# Model ve scaler yükle
model = load_model(f"model/{coin}_model.h5", compile=False)
scaler = joblib.load(f"model/{coin}_scaler.save")

# Veri çek ve işle
df = fetch_data(coin)
X, y, _ = preprocess_data(df)

# 🔮 Geçmiş veri ile tahmin
predicted = model.predict(X)

# Dummy sütunlarla birlikte ters ölçekle
dummy_cols = np.zeros((predicted.shape[0], 2))  # volume & market cap
combined = np.concatenate([predicted, dummy_cols], axis=1)
predicted_prices = scaler.inverse_transform(combined)[:, 0]

real = df["price"].values[-len(predicted):]

# 📊 Gerçek vs Tahmin (Geçmiş)
st.subheader("📊 Gerçek vs Model Tahmini (Geçmiş Veriler) - İnteraktif")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"][-len(predicted):], y=real, mode='lines', name='Gerçek'))
fig.add_trace(go.Scatter(x=df["timestamp"][-len(predicted):], y=predicted_prices, mode='lines', name='Tahmin'))
fig.update_layout(title=f"{coin.capitalize()} Fiyat Tahmini", xaxis_title="Tarih", yaxis_title="Fiyat (USD)", template="plotly_dark")
st.plotly_chart(fig)

# 📅 Gelecek Gün Tahmini
st.subheader("📅 Gelecek Gün Tahmini")

future_days = st.slider("Kaç gün ileri tahmin yapılsın?", 1, 60, 15)

# Geleceği tahmin etmek için son pencereyi al
last_window = X[-1:]
future_predictions = []

for _ in range(future_days):
    pred = model.predict(last_window, verbose=0)
    future_predictions.append(pred[0][0])

    # Yeni pencere oluştur
    next_input = np.append(last_window[0][1:], [[pred[0][0]]], axis=0)
    last_window = next_input.reshape(1, -1, 1)

# Tahminleri ters ölçekle
dummy_future = np.zeros((len(future_predictions), 2))  # volume & market_cap için
combined_future = np.column_stack([future_predictions, dummy_future])
future_prices = scaler.inverse_transform(combined_future)[:, 0]

# Gelecek tarihler
last_date = df["timestamp"].iloc[-1]
future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(future_days)]

# Plotly grafiği
fig_future = go.Figure()
fig_future.add_trace(go.Scatter(x=future_dates, y=future_prices, mode="lines+markers", name="Tahmin"))
fig_future.update_layout(title=f"{future_days} Günlük Gelecek Tahmini", xaxis_title="Tarih", yaxis_title="Fiyat (USD)", template="plotly_dark")
st.plotly_chart(fig_future)
