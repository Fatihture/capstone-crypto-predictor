import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, sequence_length=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["price", "volume", "market_cap"]])

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # sadece fiyatı tahmin etmek istiyoruz

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler
