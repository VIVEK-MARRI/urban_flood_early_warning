# airflow/dags/flood_prediction_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator 
from airflow.models import Variable 
from datetime import datetime, timedelta
import pandas as pd
import joblib
import json
import os
import shutil
import numpy as np
import mlflow
# --- FINAL CORRECTED IMPORTS ---
import mlflow.sklearn  # <-- REQUIRED for mlflow.sklearn.log_model
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
# ------------------------------

# -----------------------------
# Paths inside container
# -----------------------------
FEATURES_PATH = "models/feature_columns.json"
MODEL_PATH = "models/optimized_rf_v2.pkl"
DATA_DIR = "/opt/airflow/data"
LIVE_DATA_PATH = os.path.join(DATA_DIR, "live_data.csv")
PRED_LOG_PATH = os.path.join(DATA_DIR, "prediction_log.csv")
LAST_ROW_HASH = os.path.join(DATA_DIR, "last_row_hash.txt")
MLFLOW_ARTIFACT_BASE = "/mlflow/artifacts" # MLflow Artifact Root (Fixed)

# -----------------------------
# Configuration for Simulation
# -----------------------------
RAIN_PROB_BASE = 0.25      # Base chance of rain
MAX_RAIN_INTENSITY_LIMIT = 8.0 # Max rain value
FLOOD_THRESHOLD = 0.75     # Probability threshold to send an external alert

# -----------------------------
# Telangana major cities
# -----------------------------
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

# -----------------------------
# MLflow config
# -----------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Flood Prediction Live Logs")

# -----------------------------
# Helper functions (Keep unchanged)
# -----------------------------
def already_processed(latest_row):
    if not os.path.exists(LAST_ROW_HASH):
        return False
    try:
        comparable_data = {k: v for k, v in latest_row.items() if k not in ['minute', 'second', 'timestamp']}
        current_hash = str(hash(frozenset(comparable_data.items())))
        with open(LAST_ROW_HASH, "r") as f:
            last_hash = f.read().strip()
        return current_hash == last_hash
    except Exception:
        return False

def save_last_row_hash(latest_row):
    try:
        os.makedirs(os.path.dirname(LAST_ROW_HASH), exist_ok=True)
        comparable_data = {k: v for k, v in latest_row.items() if k not in ['minute', 'second', 'timestamp']}
        current_hash = str(hash(frozenset(comparable_data.items())))
        with open(LAST_ROW_HASH, "w") as f:
            f.write(current_hash)
    except Exception:
        pass

# -----------------------------
# Fetch / simulate live weather data (Keep fixed logic)
# -----------------------------
def fetch_weather_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    now = datetime.utcnow()

    # --- 1. Load Previous Data for Rolling Features ---
    try:
        if os.path.exists(PRED_LOG_PATH) and os.path.getsize(PRED_LOG_PATH) > 0:
            prev_log = pd.read_csv(PRED_LOG_PATH)[['rain_1h']].tail(23)
        else:
            prev_log = pd.DataFrame(columns=['rain_1h'])
    except Exception:
        prev_log = pd.DataFrame(columns=['rain_1h'])
        
    # --- 2. Generate Base Simulated Data ---
    raw_data = {
        "temp": round(np.random.uniform(20, 35), 2),
        "humidity": round(np.random.uniform(70, 100), 2), 
        "pressure": round(np.random.uniform(980, 1030), 2),
        "wind_speed": round(np.random.uniform(0, 20), 2),
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "year": now.year,
        "minute": now.minute,
        "second": now.second,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")
    }

    # --- 3. DYNAMIC RAIN SIMULATION ---
    rain_factor = (raw_data["humidity"] / 100)
    adjusted_rain_prob = RAIN_PROB_BASE * rain_factor * 1.5 

    if np.random.rand() < adjusted_rain_prob:
        raw_data["rain_1h"] = round(np.random.uniform(1, MAX_RAIN_INTENSITY_LIMIT) * rain_factor, 2)
    else:
        raw_data["rain_1h"] = 0.0

    # --- 4. Calculate Rolling Features ---
    current_rain_df = pd.DataFrame([{'rain_1h': raw_data['rain_1h']}])
    temp_df = pd.concat([prev_log, current_rain_df], ignore_index=True)
    
    raw_data['rain_3h'] = temp_df['rain_1h'].tail(3).sum()
    raw_data['rain_6h'] = temp_df['rain_1h'].tail(6).sum()
    raw_data['rain_24h'] = temp_df['rain_1h'].tail(24).sum()
    raw_data['rain_intensity'] = raw_data['rain_1h'] 
    
    # --- 5. Compute Deltas (Lagged Features) ---
    if os.path.exists(PRED_LOG_PATH) and os.path.getsize(PRED_LOG_PATH) > 0:
        try:
            prev = pd.read_csv(PRED_LOG_PATH)[['temp', 'humidity']].iloc[-1] 
            raw_data["temp_delta"] = raw_data["temp"] - prev.get("temp", raw_data["temp"])
            raw_data["humidity_delta"] = raw_data["humidity"] - prev.get("humidity", raw_data["humidity"])
        except Exception:
            raw_data["temp_delta"] = 0.0
            raw_data["humidity_delta"] = 0.0
    else:
        raw_data["temp_delta"] = 0.0
        raw_data["humidity_delta"] = 0.0

    # --- 6. Finalize and Save ---
    model_input = {k: raw_data.get(k, 0.0) for k in FEATURE_COLUMNS}
    df = pd.DataFrame([model_input], columns=FEATURE_COLUMNS)
    df.to_csv(LIVE_DATA_PATH, index=False)
    
    print(f"✅ Weather fetched. Rain (1h/24h): {raw_data['rain_1h']:.2f} / {raw_data['rain_24h']:.2f}. Humidity: {raw_data['humidity']:.1f}")
    return model_input


# -----------------------------
# Predict flood and log (FINAL CORRECTED FUNCTION)
# -----------------------------
def predict_and_log(**kwargs):
    ti = kwargs['ti']
    
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        df = pd.read_csv(LIVE_DATA_PATH)
        latest_row = df.iloc[-1].to_dict()

        if already_processed(latest_row):
            print("ℹ️ Latest row already processed. Skipping prediction.")
            ti.xcom_push(key='alert_status', value='SKIPPED') 
            ti.xcom_push(key='alert_data', value={'city': 'N/A', 'timestamp': 'N/A', 'probability': 0, 'rain_24h': 0})
            return latest_row.get('timestamp') 
        
        # --- 1. PREDICTION AND DATA PROCESSING ---
        model = joblib.load(MODEL_PATH)
        for col in FEATURE_COLUMNS:
            if col not in df.columns: df[col] = 0.0
        df = df[FEATURE_COLUMNS]
        prediction = int(model.predict(df)[0])
        probability = model.predict_proba(df)[0][list(model.classes_).index(1)]
        city = np.random.choice(list(TELANGANA_CITIES.keys()))
        base_lat, base_lon = TELANGANA_CITIES[city]
        lat = base_lat + np.random.uniform(-0.02, 0.02)
        lon = base_lon + np.random.uniform(-0.02, 0.02)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S") 

        result = {
            "prediction": prediction,
            "probability": round(float(probability), 3),
            "timestamp": timestamp,
            "lat": lat,
            "lon": lon,
            "city": city
        }

        # Save prediction log (KEEP AS IS)
        log_df = pd.DataFrame([{**df.iloc[0].to_dict(), **result}])
        if not os.path.exists(PRED_LOG_PATH):
            log_df.to_csv(PRED_LOG_PATH, index=False)
        else:
            log_df.to_csv(PRED_LOG_PATH, mode='a', index=False, header=False)
        
        # Clean duplicates and sort (KEEP AS IS)
        full_log = pd.read_csv(PRED_LOG_PATH)
        full_log['timestamp'] = pd.to_datetime(full_log['timestamp'], errors='coerce')
        full_log.sort_values(by='timestamp', inplace=True)
        full_log.drop_duplicates(subset=['timestamp', 'city'], keep='last', inplace=True)
        full_log.to_csv(PRED_LOG_PATH, index=False)
        save_last_row_hash(latest_row)

        # -----------------------------
        # --- MLFLOW LOGGING (FINAL CORRECTED BLOCK) ---
        # -----------------------------
        try:
            with mlflow.start_run(run_name=f"LivePrediction_{timestamp}") as run:
                mlflow.log_params(df.iloc[0].to_dict())
                mlflow.log_metric("flood_probability", probability)
                mlflow.log_metric("flood_prediction", prediction)
                
                # Logs the model used for full reproducibility (FIX)
                mlflow.sklearn.log_model(model, artifact_path="prediction_model") 
                
                # Logs the input data used for this specific prediction (FIX)
                mlflow.log_artifact(LIVE_DATA_PATH, artifact_path="live_input_data")
                
                print(f"✅ MLflow run logged: {run.info.run_id}")
        except Exception as ml_err:
            # Logs MLflow failures but allows the rest of the DAG to complete
            print("❌ MLflow logging failed:", ml_err)
            traceback.print_exc()
        # -----------------------------
        # --- MLFLOW LOGGING (END) ---
        # -----------------------------
        
        # --- FINAL XCOM PUSH & ALERT STATUS CHECK ---
        alert_data = {
            "prediction": prediction,
            "probability": round(float(probability), 3),
            "timestamp": timestamp,
            "city": city,
            "rain_24h": df.iloc[0].get('rain_24h')
        }
        
        if probability >= FLOOD_THRESHOLD:
            ti.xcom_push(key='alert_status', value='HIGH_RISK')
        else:
            ti.xcom_push(key='alert_status', value='LOW_RISK')
            
        print("✅ Prediction logged:", result)
        ti.xcom_push(key='alert_data', value=alert_data) 
        return "Prediction_Completed"

    except Exception as e:
        # Catches major errors (e.g., file not found, model load failed)
        print("❌ Prediction failed:", e)
        traceback.print_exc()
        ti.xcom_push(key='alert_status', value='ERROR')
        ti.xcom_push(key='alert_data', value={'city': 'N/A (Error)', 'timestamp': datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S"), 'probability': 0, 'rain_24h': 0, 'error': str(e)})
        return "Prediction_Failed"

# -----------------------------
# NEW ALERTING TASK DEFINITIONS (Keep fixed logic)
# -----------------------------
def check_flood_risk_callable(**kwargs):
    """Determines whether to send a Slack alert based on the XCom status."""
    ti = kwargs['ti']
    status = ti.xcom_pull(task_ids='predict_and_log', key='alert_status')
    
    if status == 'HIGH_RISK':
        return 'send_slack_alert'
    else:
        return 'no_alert_needed' 
        
def create_slack_alert_tasks(dag):
    """Defines the branching and slack notification tasks."""
    
    # --- 1. Branching Task ---
    risk_branch = PythonOperator(
        task_id='check_flood_risk',
        python_callable=check_flood_risk_callable,
        provide_context=True,
        dag=dag,
    )
    
    # --- 2. Alert Task (FINAL FIX FOR TEMPLATE SYNTAX) ---
    slack_message = f"""
    {{% set data = ti.xcom_pull(task_ids='predict_and_log', key='alert_data', default='{{"city": "N/A (Skipped)", "timestamp": "N/A", "probability": 0, "rain_24h": 0}}') %}}
    :rotating_light: *URGENT FLOOD ALERT* :rotating_light:
    
    *Location:* {{{{ data['city'] }}}}
    *Time:* {{{{ data['timestamp'] }}}}
    *Predicted Probability:* *{{{{ data['probability'] | float | round(2) }}}}* *Rain (24h Accumulated):* {{{{ data['rain_24h'] | float | round(2) }}}} units
    
    > :warning: *ACTION REQUIRED:* Flood risk has surpassed the {FLOOD_THRESHOLD*100:.0f}% threshold.
    """
    
    send_alert = SlackWebhookOperator(
        task_id='send_slack_alert',
        slack_webhook_conn_id='slack_webhook_default', 
        message=slack_message,
        dag=dag,
    )
    
    # --- 3. No-Op Task (for clean branching) ---
    no_op = EmptyOperator(
        task_id='no_alert_needed',
        dag=dag,
    )
    
    return risk_branch, send_alert, no_op

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
    description="Automated flood prediction pipeline with MLflow logging and conditional Slack alerting",
    schedule_interval="*/5 * * * *",
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
    provide_context=True, # Necessary for XCom push/pull
    dag=dag
)

# Create the alerting branch tasks
risk_branch, send_alert, no_op = create_slack_alert_tasks(dag)

# Define the flow: Fetch -> Predict -> Branch/Alert
fetch_task >> predict_task >> risk_branch
risk_branch >> [send_alert, no_op]