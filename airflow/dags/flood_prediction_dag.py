"" 
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import joblib
import random
import os

# Define model and data paths
MODEL_PATH = os.path.abspath("../../models/random_forest_baseline.pkl")
DATA_DIR = os.path.abspath("../../data")

def fetch_weather_data():
    """Simulate fetching live weather data"""
    os.makedirs(DATA_DIR, exist_ok=True)
    data = {
        "rain_1h": random.uniform(0, 30),
        "rain_3h": random.uniform(0, 80),
        "temperature": random.uniform(20, 35),
        "humidity": random.uniform(60, 100),
        "wind_speed": random.uniform(0, 20)
    }
    df = pd.DataFrame([data])
    df.to_csv(os.path.join(DATA_DIR, "live_data.csv"), index=False)
    print("✅ Weather data fetched:", data)
    return data

def predict_flood():
    """Predict flood risk using trained model"""
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(os.path.join(DATA_DIR, "live_data.csv"))
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    result = {"prediction": int(prediction), "probability": round(prob, 2)}

    log_df = pd.DataFrame([{**df.iloc[0].to_dict(), **result, "timestamp": datetime.now()}])
    log_path = os.path.join(DATA_DIR, "prediction_log.csv")
    log_df.to_csv(log_path, mode='a', index=False, header=not os.path.exists(log_path))

    print("✅ Prediction complete:", result)
    return result

# DAG setup
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 10, 14),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    "flood_prediction_pipeline",
    default_args=default_args,
    description="Automated flood prediction pipeline",
    schedule_interval="*/5 * * * *",  # every 5 minutes
    catchup=False,
)

fetch_task = PythonOperator(task_id="fetch_weather_data", python_callable=fetch_weather_data, dag=dag)
predict_task = PythonOperator(task_id="predict_flood", python_callable=predict_flood, dag=dag)

fetch_task >> predict_task
