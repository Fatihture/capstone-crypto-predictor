import requests
import pandas as pd

def fetch_data(coin_id="bitcoin", days="365"):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if "prices" not in data or "total_volumes" not in data or "market_caps" not in data:
        raise ValueError(f"API'den eksik veri geldi: {data}")

    prices = data["prices"]
    volumes = data["total_volumes"]
    market_caps = data["market_caps"]
    
    df = pd.DataFrame({
        "timestamp": [x[0] for x in prices],
        "price": [x[1] for x in prices],
        "volume": [x[1] for x in volumes],
        "market_cap": [x[1] for x in market_caps]
    })
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
