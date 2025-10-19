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
import shutil
import numpy as np

# -----------------------------
# Paths inside container
# -----------------------------
FEATURES_PATH = "/opt/airflow/models/feature_columns.json"
MODEL_PATH = "/opt/airflow/models/optimized_rf_model.pkl"
DATA_DIR = "/opt/airflow/data"
LIVE_DATA_PATH = os.path.join(DATA_DIR, "live_data.csv")
PRED_LOG_PATH = os.path.join(DATA_DIR, "prediction_log.csv")
ARTIFACT_FOLDER = "/mlflow/artifacts/live_run"
LAST_ROW_HASH = os.path.join(DATA_DIR, "last_row_hash.txt")

# Telangana major cities
TELANGANA_CITIES = {
    "Hyderabad": (17.3850, 78.4867),
    "Warangal": (17.9789, 79.5915),
    "Nizamabad": (18.6720, 78.0941),
    "Karimnagar": (18.4386, 79.1288),
    "Khammam": (17.2473, 80.1514),
    "Mahbubnagar": (16.7435, 78.0081),
    "Suryapet": (17.1500, 79.6167),
    "Adilabad": (19.6667, 78.5333)
}

# -----------------------------
# Load feature columns
# -----------------------------
with open(FEATURES_PATH, "r") as f:
    FEATURE_COLUMNS = json.load(f)

DEFAULT_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# -----------------------------
# Helper: check if latest row was already processed
# -----------------------------
def already_processed(latest_row):
    if not os.path.exists(LAST_ROW_HASH):
        return False
    try:
        current_hash = str(hash(frozenset(latest_row.items())))
        with open(LAST_ROW_HASH, "r") as f:
            last_hash = f.read().strip()
        return current_hash == last_hash
    except Exception:
        return False

def save_last_row_hash(latest_row):
    try:
        os.makedirs(os.path.dirname(LAST_ROW_HASH), exist_ok=True)
        current_hash = str(hash(frozenset(latest_row.items())))
        with open(LAST_ROW_HASH, "w") as f:
            f.write(current_hash)
    except Exception:
        pass

# -----------------------------
# Fetch / simulate weather data
# -----------------------------
def fetch_weather_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    now = datetime.utcnow()

    raw_data = {
        "temp": round(random.uniform(20, 35), 2),
        "humidity": round(random.uniform(60, 100), 2),
        "pressure": round(random.uniform(980, 1050), 2),
        "wind_speed": round(random.uniform(0, 20), 2),
        "rain_1h": round(random.uniform(0, 30), 2),
        "rain_3h": round(random.uniform(0, 80), 2),
        "rain_6h": round(random.uniform(0, 150), 2),
        "rain_24h": round(random.uniform(0, 300), 2),
        "rain_intensity": round(random.uniform(0, 10), 2),
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second,
        "day": now.day,
        "month": now.month,
        "year": now.year
    }

    if os.path.exists(PRED_LOG_PATH):
        try:
            prev = pd.read_csv(PRED_LOG_PATH).iloc[-1]
            raw_data["temp_delta"] = raw_data["temp"] - prev.get("temp", raw_data["temp"])
            raw_data["humidity_delta"] = raw_data["humidity"] - prev.get("humidity", raw_data["humidity"])
        except Exception:
            raw_data["temp_delta"] = 0.0
            raw_data["humidity_delta"] = 0.0
    else:
        raw_data["temp_delta"] = 0.0
        raw_data["humidity_delta"] = 0.0

    model_input = {k: raw_data.get(k, 0.0) for k in FEATURE_COLUMNS}
    df = pd.DataFrame([model_input], columns=FEATURE_COLUMNS)
    df.to_csv(LIVE_DATA_PATH, index=False)
    print("✅ Weather data fetched:", model_input)
    return model_input

# -----------------------------
# Predict flood and log
# -----------------------------
def predict_and_log():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_FOLDER, exist_ok=True)
    mlflow.set_tracking_uri(DEFAULT_MLFLOW_URI)
    mlflow.set_experiment("Flood Prediction Live Logs")

    try:
        df = pd.read_csv(LIVE_DATA_PATH)
        latest_row = df.iloc[-1].to_dict()
        if already_processed(latest_row):
            print("ℹ️ Latest row already processed. Skipping prediction.")
            return {"status": "skipped"}

        model = joblib.load(MODEL_PATH)
        prediction = int(model.predict(df)[0])
        probability = 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)
            positive_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
            probability = float(proba[0][positive_index])

        city = np.random.choice(list(TELANGANA_CITIES.keys()))
        base_lat, base_lon = TELANGANA_CITIES[city]
        lat = base_lat + np.random.uniform(-0.02, 0.02)
        lon = base_lon + np.random.uniform(-0.02, 0.02)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        result = {
            "prediction": prediction,
            "probability": round(probability, 3),
            "timestamp": timestamp,
            "lat": lat,
            "lon": lon,
            "city": city
        }

        log_df = pd.DataFrame([{**df.iloc[0].to_dict(), **result}])
        if not os.path.exists(PRED_LOG_PATH):
            log_df.to_csv(PRED_LOG_PATH, index=False)
        else:
            log_df.to_csv(PRED_LOG_PATH, mode='a', index=False, header=False)

        # -----------------------------
        # Ensure chronological order & remove duplicates
        # -----------------------------
        full_log = pd.read_csv(PRED_LOG_PATH)
        full_log['timestamp'] = pd.to_datetime(full_log['timestamp'], errors='coerce')
        full_log.sort_values(by='timestamp', inplace=True)
        full_log.drop_duplicates(subset=['timestamp','city'], keep='last', inplace=True)
        full_log.to_csv(PRED_LOG_PATH, index=False)

        save_last_row_hash(latest_row)

        shutil.copy(LIVE_DATA_PATH, ARTIFACT_FOLDER)
        shutil.copy(MODEL_PATH, ARTIFACT_FOLDER)

        with mlflow.start_run(run_name="Live Flood Prediction"):
            mlflow.log_params(df.iloc[0].to_dict())
            mlflow.log_metric("flood_probability", probability)
            mlflow.log_metric("flood_prediction", prediction)
            mlflow.log_artifacts(ARTIFACT_FOLDER)

        print("✅ Prediction logged:", result)
        return result

    except Exception as e:
        print("❌ Prediction failed:", e)
        return {"error": str(e)}

# -----------------------------
# DAG setup
# -----------------------------
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
