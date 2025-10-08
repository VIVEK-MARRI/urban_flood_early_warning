import time
import schedule
from src.data_ingestion import fetch_weather, save_to_csv

def job():
    print("Fetching new weather data...")
    data = fetch_weather()
    save_to_csv(data)
    print("✅ Data saved successfully!\n")

# Schedule job every hour
schedule.every(1).hours.do(job)

if __name__ == "__main__":
    print("⏳ Starting data collection service (runs every hour)...")
    job()  # Run once immediately
    while True:
        schedule.run_pending()
        time.sleep(60)
