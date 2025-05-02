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
dummy_cols = np.zeros((predicted.shape[0], 2))  # 2 ekstra sütun
combined = np.concatenate([predicted, dummy_cols], axis=1)
predicted_prices = scaler.inverse_transform(combined)[:, 0]

real = df["price"].values[-len(predicted):]

# 📊 Gerçek vs Tahmin (Plotly ile interaktif grafik)
st.subheader("📊 Gerçek vs Model Tahmini (Geçmiş Veriler) - İnteraktif")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"][-len(predicted):], y=real, mode='lines', name='Gerçek'))
fig.add_trace(go.Scatter(x=df["timestamp"][-len(predicted):], y=predicted_prices, mode='lines', name='Tahmin'))
fig.update_layout(title=f"{coin.capitalize()} Fiyat Tahmini", xaxis_title="Tarih", yaxis_title="Fiyat (USD)", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# 📉 Hata Metrikleri
rmse = np.sqrt(mean_squared_error(real, predicted_prices))
mae = mean_absolute_error(real, predicted_prices)
mape = np.mean(np.abs((real - predicted_prices) / real)) * 100

st.markdown("### 📊 Model Performansı (Geçmiş Veriler)")
st.write(f"**RMSE:** {rmse:,.2f}")
st.write(f"**MAE:** {mae:,.2f}")
st.write(f"**MAPE:** %{mape:.2f}")

# 🔮 7 Günlük İleri Tahmin Fonksiyonu   "recursive forecasting"
def predict_next_days(model, last_sequence, days, scaler):
    predictions = []
    current_seq = last_sequence.copy()
    for _ in range(days):
        pred = model.predict(current_seq[np.newaxis, :, :])[0]
        pred_full = np.concatenate([pred, [0, 0]])  # (3,)
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [pred_full], axis=0)
    predictions = np.array(predictions)

    # Dummy sütun ekle (volume ve market_cap)
    dummy_cols = np.zeros((predictions.shape[0], 2))
    combined = np.concatenate([predictions, dummy_cols], axis=1)
    return scaler.inverse_transform(combined)[:, 0]

# 🔁 Son diziyi alıp ileriye tahmin et
future_days = 7
last_seq = X[-1]
future_predictions = predict_next_days(model, last_seq, future_days, scaler)

# 📅 7 Günlük Tahmin Grafiği
st.subheader("📅 7 Günlük İleriye Dönük Tahmin")
future_dates = [df["timestamp"].iloc[-1] + datetime.timedelta(days=i+1) for i in range(future_days)]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name="İleri Tahmin", line=dict(color='orange')))
fig2.update_layout(title="📈 Gelecek 7 Günlük Fiyat Tahmini", xaxis_title="Tarih", yaxis_title="Fiyat (USD)", template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# 📋 Tablo + İndirme
future_df = pd.DataFrame({
    "Tarih": future_dates,
    "Tahmin Edilen Fiyat (USD)": future_predictions
})
st.markdown("### 📋 Tahmin Tablosu (7 Günlük)")
st.dataframe(future_df)

csv = future_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ CSV olarak indir",
    data=csv,
    file_name=f'{coin}_7_gunluk_tahmin.csv',
    mime='text/csv',
)

# python -m streamlit run app/streamlit_app.py