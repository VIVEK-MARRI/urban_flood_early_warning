# airflow/dags/flood_prediction_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import joblib
import random
import mlflow
import os
import json

# Paths inside container
FEATURES_PATH = "/opt/airflow/models/feature_columns.json"
MODEL_PATH = "/opt/airflow/models/random_forest_baseline.pkl"
DATA_DIR = "/opt/airflow/data"
LIVE_DATA_PATH = os.path.join(DATA_DIR, "live_data.csv")
PRED_LOG_PATH = os.path.join(DATA_DIR, "prediction_log.csv")

# Load feature columns
with open(FEATURES_PATH, "r") as f:
    FEATURE_COLUMNS = json.load(f)

# MLflow setup
DEFAULT_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def fetch_weather_data():
    """Fetch or simulate live weather data and compute delta/time features."""
    os.makedirs(DATA_DIR, exist_ok=True)
    now = datetime.now()

    raw_data = {
        "temp": random.uniform(20, 35),
        "humidity": random.uniform(60, 100),
        "pressure": random.uniform(980, 1050),
        "wind_speed": random.uniform(0, 20),
        "rain_1h": random.uniform(0, 30),
        "rain_3h": random.uniform(0, 80),
        "rain_6h": random.uniform(0, 150),
        "rain_24h": random.uniform(0, 300),
        "rain_intensity": random.uniform(0, 10),
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second,
        "day": now.day,
        "month": now.month,
        "year": now.year
    }

    # Compute deltas from previous log
    if os.path.exists(PRED_LOG_PATH):
        prev = pd.read_csv(PRED_LOG_PATH).iloc[-1]
        raw_data["temp_delta"] = raw_data["temp"] - prev.get("temp", raw_data["temp"])
        raw_data["humidity_delta"] = raw_data["humidity"] - prev.get("humidity", raw_data["humidity"])
    else:
        raw_data["temp_delta"] = 0.0
        raw_data["humidity_delta"] = 0.0

    # Ensure all features exist in correct order
    model_input = {k: raw_data.get(k, 0.0) for k in FEATURE_COLUMNS}
    df = pd.DataFrame([model_input], columns=FEATURE_COLUMNS)

    df.to_csv(LIVE_DATA_PATH, index=False)
    print("✅ Weather data fetched:", model_input)
    return model_input


def predict_and_log():
    """Predict flood and log results to CSV + MLflow."""
    os.makedirs(DATA_DIR, exist_ok=True)
    mlflow.set_tracking_uri(DEFAULT_MLFLOW_URI)
    mlflow.set_experiment("Flood Prediction Live Logs")

    try:
        # Load model and live data
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(LIVE_DATA_PATH)

        # Predict class
        prediction = int(model.predict(df)[0])

        # Handle probability safely
        proba = model.predict_proba(df)
        if proba.shape[1] == 1:
            probability = float(proba[0][0])
        else:
            # Pick probability for positive (flood) class if available
            positive_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
            probability = float(proba[0][positive_index])

        result = {
            "prediction": prediction,
            "probability": round(probability, 3),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Log to CSV
        log_df = pd.DataFrame([{**df.iloc[0].to_dict(), **result}])
        if not os.path.exists(PRED_LOG_PATH):
            log_df.to_csv(PRED_LOG_PATH, index=False)
        else:
            log_df.to_csv(PRED_LOG_PATH, mode='a', index=False, header=False)

        # Log to MLflow
        with mlflow.start_run(run_name="Live Flood Prediction"):
            mlflow.log_params(df.iloc[0].to_dict())
            mlflow.log_metric("flood_probability", probability)
            mlflow.log_metric("flood_prediction", prediction)
            try:
                mlflow.log_artifact(LIVE_DATA_PATH)
                mlflow.log_artifact(MODEL_PATH)
            except Exception as e:
                print("⚠️ Warning: could not log artifact to MLflow:", e)

        print("✅ Prediction logged:", result)
        return result

    except Exception as e:
        print("❌ Prediction failed:", e)
        return {"error": str(e)}


# DAG setup
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 10, 15),
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

fetch_task = PythonOperator(
    task_id="fetch_weather_data",
    python_callable=fetch_weather_data,
    dag=dag
)

predict_task = PythonOperator(
    task_id="predict_and_log",
    python_callable=predict_and_log,
    dag=dag
)

fetch_task >> predict_task
