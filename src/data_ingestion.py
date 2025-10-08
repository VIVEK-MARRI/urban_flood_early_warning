# src/data_ingestion.py
import requests
import pandas as pd
from datetime import datetime
import os

# ---- CONFIG ----
API_KEY = "52ca078b59c3685cb762cbc7dd9ad5fa"
CITY = "Hyderabad,in"
DATA_PATH = "data/flood_data.csv"

# ---- FUNCTION TO FETCH WEATHER ----
def fetch_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    # Extract relevant fields
    weather_data = {
        "timestamp": datetime.now(),
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "rain_1h": data.get("rain", {}).get("1h", 0)
    }
    return weather_data

# ---- SAVE DATA TO CSV ----
def save_to_csv(data):
    if not os.path.exists(DATA_PATH):
        df = pd.DataFrame([data])
    else:
        df = pd.read_csv(DATA_PATH)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)


# ---- MAIN ----
if __name__ == "__main__":
    weather = fetch_weather()
    save_to_csv(weather)
    print("Weather data saved:", weather)
