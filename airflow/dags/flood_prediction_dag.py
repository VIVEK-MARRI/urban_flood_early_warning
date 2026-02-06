# airflow/dags/flood_prediction_pipeline.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import json # Explicitly import json
import hashlib # For stable hashing
import joblib
import mlflow
import traceback

# -----------------------------
# Paths and Config
# -----------------------------
FEATURES_PATH = "models/feature_columns.json"
MODEL_PATH = "models/urban_flood_model.pkl"
DATA_DIR = "/opt/airflow/data"
LIVE_DATA_PATH = os.path.join(DATA_DIR, "live_data.csv")
PRED_LOG_PATH = os.path.join(DATA_DIR, "prediction_log.csv")
LAST_ROW_HASH = os.path.join(DATA_DIR, "last_row_hash.txt")
RAIN_PROB_BASE = 0.25
MAX_RAIN_INTENSITY_LIMIT = 8.0
FLOOD_THRESHOLD = 0.75
FORCE_RAIN_INTERVAL = 10

# -----------------------------
# REQUIRED GLOBAL DEFINITIONS
# -----------------------------
TELANGANA_CITIES = {
    "Hyderabad": (17.3850, 78.4867),
    "Warangal": (17.9789, 79.5915),
    "Nizamabad": (18.6720, 78.0941),
    "Karimnagar": (18.4386, 79.1288),
    "Khammam": (17.2473, 80.1514),
    "Mahbubnagar": (16.7435, 78.0081),
    "Suryapet": (17.1500, 79.6167),
    "Adilabad": (19.6667, 78.5333),
    "Ramagundam": (18.7750, 79.4500),
    "Secunderabad": (17.4399, 78.4983),
    "Kothagudem": (17.9392, 80.3150),
    "Sangareddy": (17.8444, 78.0833),
    "Siddipet": (18.1064, 78.8528),
    "Jagtial": (18.7900, 78.9200),
    "Mancherial": (18.8789, 79.4678),
    "Peddapalli": (18.6250, 79.4000),
    "Vikarabad": (17.3333, 77.9167),
    "Bhongir": (17.5144, 78.8953),
    "Kamareddy": (18.3100, 78.3400),
}


# -----------------------------
# Load feature columns (Global scope for safe function access)
# -----------------------------
try:
    with open(FEATURES_PATH, "r") as f:
        FEATURE_COLUMNS = json.load(f)
except Exception:
    # Fallback list must be declared in global scope
    FEATURE_COLUMNS = [
        "temp", "humidity", "pressure", "wind_speed", "rain_1h", "rain_3h",
        "rain_6h", "rain_24h", "rain_intensity", "temp_delta", "humidity_delta",
        "hour", "day", "month", "year"
    ]

# -----------------------------
# MLflow setup
# -----------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Flood Prediction Live Logs")

# -----------------------------
# Helper functions (Assumed correct and available)
# -----------------------------
def already_processed(latest_row):
    if not os.path.exists(LAST_ROW_HASH):
        return False
    try:
        comparable_data = {k: v for k, v in latest_row.items() if k not in ['minute', 'second', 'timestamp']}
        # Use stable SHA-256 hash instead of built-in hash()
        row_str = json.dumps(comparable_data, sort_keys=True)
        current_hash = hashlib.sha256(row_str.encode('utf-8')).hexdigest()
        
        with open(LAST_ROW_HASH, "r") as f:
            last_hash = f.read().strip()
        return current_hash == last_hash
    except Exception:
        return False


def save_last_row_hash(latest_row):
    try:
        os.makedirs(os.path.dirname(LAST_ROW_HASH), exist_ok=True)
        comparable_data = {k: v for k, v in latest_row.items() if k not in ['minute', 'second', 'timestamp']}
        # Use stable SHA-256 hash
        row_str = json.dumps(comparable_data, sort_keys=True)
        current_hash = hashlib.sha256(row_str.encode('utf-8')).hexdigest()
        
        with open(LAST_ROW_HASH, "w") as f:
            f.write(current_hash)
    except Exception:
        pass


# -----------------------------
# Fetch simulated live data
# -----------------------------
def fetch_weather_data(**kwargs):
    os.makedirs(DATA_DIR, exist_ok=True)
    now = datetime.utcnow()

    # --- 1. Load Previous Data (Reading from old log for history features) ---
    try:
        if os.path.exists(PRED_LOG_PATH) and os.path.getsize(PRED_LOG_PATH) > 0:
            full_log = pd.read_csv(PRED_LOG_PATH)
            full_log['timestamp'] = pd.to_datetime(full_log['timestamp'], errors='coerce')
            full_log.sort_values(by='timestamp', inplace=True)
            prev_log = full_log[['rain_1h', 'temp', 'humidity']].tail(23).reset_index(drop=True)
        else:
            prev_log = pd.DataFrame(columns=['rain_1h', 'temp', 'humidity'])
    except Exception:
        prev_log = pd.DataFrame(columns=['rain_1h', 'temp', 'humidity'])

    # --- 2. Get Run Count for Forced Alert ---
    run_count = int(Variable.get("flood_run_count", default_var=0))
    run_count += 1
    Variable.set("flood_run_count", run_count)
    is_force_rain = (run_count % FORCE_RAIN_INTERVAL == 0)

    # --- 3. Generate Base Data ---
    raw_data = {
        "temp": round(np.random.uniform(20, 35), 2),
        "humidity": round(np.random.uniform(70, 100), 2),
        "pressure": round(np.random.uniform(980, 1030), 2),
        "wind_speed": round(np.random.uniform(0, 20), 2),
        "hour": now.hour, "day": now.day, "month": now.month, "year": now.year,
        "minute": now.minute, "second": now.second,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")
    }

    # --- 4. DYNAMIC RAIN SIMULATION (Fixed logic) ---
    rain_factor = raw_data["humidity"] / 100
    adjusted_rain_prob = RAIN_PROB_BASE * rain_factor * 1.5

    if is_force_rain:
        raw_data["rain_1h"] = round(np.random.uniform(5.0, MAX_RAIN_INTENSITY_LIMIT), 2)
        print(f"🚨 Forced rain event: {raw_data['rain_1h']} mm/h (Run {run_count})")
    elif np.random.rand() < adjusted_rain_prob:
        raw_data["rain_1h"] = round(np.random.uniform(1, MAX_RAIN_INTENSITY_LIMIT) * rain_factor, 2)
    else:
        raw_data["rain_1h"] = 0.0

    # --- 5. Calculate Rolling Features ---
    current_rain_df = pd.DataFrame([{'rain_1h': raw_data['rain_1h']}])
    temp_df = pd.concat([prev_log[['rain_1h']], current_rain_df], ignore_index=True)

    raw_data['rain_3h'] = temp_df['rain_1h'].tail(3).sum()
    raw_data['rain_6h'] = temp_df['rain_1h'].tail(6).sum()
    raw_data['rain_24h'] = temp_df['rain_1h'].tail(24).sum()
    raw_data['rain_intensity'] = raw_data['rain_1h']

    # --- 6. Compute Deltas (Lagged Features) ---
    if not prev_log.empty:
        prev = prev_log.iloc[-1]
        raw_data["temp_delta"] = raw_data["temp"] - prev.get("temp", raw_data["temp"])
        raw_data["humidity_delta"] = raw_data["humidity"] - prev.get("humidity", raw_data["humidity"])
    else:
        raw_data["temp_delta"] = 0.0
        raw_data["humidity_delta"] = 0.0

    # --- 7. Finalize and Save to LIVE_DATA_PATH ---
    # FEATURE_COLUMNS is now safely accessible from the global scope
    model_input = {k: raw_data.get(k, 0.0) for k in FEATURE_COLUMNS} 
    df = pd.DataFrame([model_input], columns=FEATURE_COLUMNS)
    df.to_csv(LIVE_DATA_PATH, index=False)

    print(f"✅ Weather fetched. Rain (1h/24h): {raw_data['rain_1h']} / {raw_data['rain_24h']}")
    return model_input


# -----------------------------
# Predict Flood
# -----------------------------
def predict_and_log(**kwargs):
    ti = kwargs['ti']
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        df = pd.read_csv(LIVE_DATA_PATH)
        latest_row = df.iloc[-1].to_dict()

        if already_processed(latest_row):
            print("ℹ️ Row already processed.")
            ti.xcom_push(key='prediction_log_data', value=None)
            ti.xcom_push(key='alert_status', value='SKIPPED')
            return

        model = joblib.load(MODEL_PATH)
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        df = df[FEATURE_COLUMNS]
        proba_output = model.predict_proba(df)
        probability = proba_output[0][list(model.classes_).index(1)]
        prediction = int(probability >= FLOOD_THRESHOLD)

        # NOTE: TELANGANA_CITIES is now safely accessible from the global scope
        city = np.random.choice(list(TELANGANA_CITIES.keys())) 
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        log_data = {
            "timestamp": timestamp,
            "city": city,
            "prediction": prediction,
            "probability": round(float(probability), 3),
            "rain_24h": df.iloc[0].get('rain_24h'),
            "temp": df.iloc[0].get('temp'),
            "humidity": df.iloc[0].get('humidity'),
            "pressure": df.iloc[0].get('pressure'),
        }

        # MLflow logging
        try:
            with mlflow.start_run(run_name=f"LivePrediction_{timestamp}") as run:
                mlflow.log_params(df.iloc[0].to_dict())
                mlflow.log_metric("flood_probability", probability)
                mlflow.log_param("model_path", MODEL_PATH)
                mlflow.log_artifact(LIVE_DATA_PATH, artifact_path="live_input_data")
        except Exception as e:
            print(f"MLflow logging error: {e}")

        # XCom pushes
        ti.xcom_push(key='prediction_log_data', value=log_data)
        ti.xcom_push(key='alert_data', value=log_data)
        ti.xcom_push(key='alert_status', value='HIGH_RISK' if probability >= FLOOD_THRESHOLD else 'LOW_RISK')

        save_last_row_hash(latest_row)
        print("✅ Prediction generated and pushed for logging.")
        return "Prediction_Completed"

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        ti.xcom_push(key='prediction_log_data', value=None)
        ti.xcom_push(key='alert_status', value='ERROR')
        return "Prediction_Failed"


# -----------------------------
# Log to PostgreSQL (Function remains unchanged)
# -----------------------------
def log_to_postgres(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='predict_and_log', key='prediction_log_data')
    if not data:
        print("ℹ️ No data to log.")
        return

    # NOTE: Uses the 'postgres_default' connection defined by airflow-init
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    table_name = 'flood_predictions_log'

    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP UNIQUE,
            city VARCHAR(50),
            prediction INT,
            probability FLOAT,
            rain_24h FLOAT,
            temp FLOAT,
            humidity FLOAT,
            pressure FLOAT
        );
    """

    cols = ['timestamp', 'city', 'prediction', 'probability', 'rain_24h', 'temp', 'humidity', 'pressure']
    values = [data[c] for c in cols]
    insert_stmt = f"""
        INSERT INTO {table_name} ({', '.join(cols)})
        VALUES ({', '.join(['%s'] * len(values))})
        ON CONFLICT (timestamp) DO NOTHING
    """

    try:
        pg_hook.run(create_table_sql)
        pg_hook.run(insert_stmt, parameters=values)
        print(f"✅ Logged prediction for {data['city']} to PostgreSQL.")
    except Exception as e:
        print(f"❌ Failed to log prediction to PostgreSQL. Error: {e}")
        # Note: We raise here to fail the task and signal downstream tasks
        raise


# -----------------------------
# Flood Alert Logic + Slack (Functions remain unchanged)
# -----------------------------
def check_flood_risk_callable(**kwargs):
    ti = kwargs['ti']
    status = ti.xcom_pull(task_ids='predict_and_log', key='alert_status')
    return 'send_slack_alert' if status == 'HIGH_RISK' else 'no_alert_needed'


def create_slack_alert_tasks(dag):
    FLOOD_THRESHOLD = 0.75 # Local definition for template usage
    
    risk_branch = PythonOperator(
        task_id='check_flood_risk',
        python_callable=check_flood_risk_callable,
        provide_context=True,
        dag=dag,
    )

    slack_message = f"""
    {{% set data = ti.xcom_pull(task_ids='predict_and_log', key='alert_data', default={{"city": "N/A", "timestamp": "N/A", "probability": 0, "rain_24h": 0}}) %}}
    :rotating_light: *URGENT FLOOD ALERT* :rotating_light:
    *Location:* {{{{ data['city'] }}}}
    *Time:* {{{{ data['timestamp'] }}}}
    *Predicted Probability:* *{{{{ data['probability'] | float | round(2) }}}}*
    *Rain (24h):* {{{{ data['rain_24h'] | float | round(2) }}}} mm
    """

    send_alert = SlackWebhookOperator(
        task_id='send_slack_alert',
        slack_webhook_conn_id='slack_webhook_default',
        message=slack_message,
        dag=dag,
    )

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
    description="Automated flood prediction DAG with MLflow logging, DB audit, and Slack alerts",
    schedule_interval="*/5 * * * *",
    catchup=False,
)

fetch_task = PythonOperator(task_id="fetch_weather_data", python_callable=fetch_weather_data, provide_context=True, dag=dag)
predict_task = PythonOperator(task_id="predict_and_log", python_callable=predict_and_log, provide_context=True, dag=dag)
log_db_task = PythonOperator(task_id="log_to_database", python_callable=log_to_postgres, provide_context=True, dag=dag)

risk_branch, send_alert, no_op = create_slack_alert_tasks(dag)

# Corrected DAG Flow: Fetch -> Predict -> Log -> Risk Branch
fetch_task >> predict_task >> log_db_task >> risk_branch
risk_branch >> [send_alert, no_op]
