import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
import joblib
from utils.fetch_data import fetch_data
from utils.preprocess import preprocess_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
import os

def build_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_and_save_model(coin_id):
    df = fetch_data(coin_id)
    X, y, scaler = preprocess_data(df)

    model = build_model((X.shape[1], X.shape[2]))

    model.fit(X, y, epochs=30, batch_size=16, validation_split=0.2,  # modeli eğittim
              callbacks=[EarlyStopping(patience=5)], verbose=1)

    model.save(f"model/{coin_id}_model.h5")   #h5 olarak kaydettim
    joblib.dump(scaler, f"model/{coin_id}_scaler.save")

if __name__ == "__main__":
    for coin in ["bitcoin", "ethereum", "solana"]:  #burda 3ü için de model eğitiliyor, kaydediliyor
        train_and_save_model(coin)
