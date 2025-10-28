# airflow/dags/retrain_logic.py

import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, recall_score, brier_score_loss

# --- Configuration ---
RAW_LOG_PATH = "data/prediction_log.csv"
MODEL_NAME = "UrbanFloodClassifier" 
TARGET_COLUMN = 'flood_risk'
METRICS_THRESHOLD = 0.95 # Minimum F1 score required for promotion

# --- Feature Engineering and Labeling Logic ---
def create_flood_label(df):
    """Recreates the flood risk proxy label used in initial training."""
    # Ensure columns exist; df should contain 'rain_24h', 'humidity', 'pressure' from logs
    df['rain_24h_prev'] = df['rain_24h'].shift(1)
    df['humidity_prev'] = df['humidity'].shift(1)
    df['pressure_prev'] = df['pressure'].shift(1)

    # Use a set of conditions that define a high-risk event based on historical features
    label = (
        (df['rain_24h_prev'] >= 1.8) &  # Example threshold: substantial 24h rain accumulation
        (df['humidity_prev'] >= 75.0) & # Example threshold: high humidity
        (df['pressure_prev'] <= 1012.0) # Example threshold: low pressure
    ).astype(int)
    
    # Clean up proxy columns used for lag calculation
    df.drop(columns=['rain_24h_prev', 'humidity_prev', 'pressure_prev'], inplace=True, errors='ignore')
    return label

def preprocess_and_split(df):
    """Prepares features, derives labels, and splits data for training/validation."""
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # Recalculate Flood Risk Label based on the latest log data
    df[TARGET_COLUMN] = create_flood_label(df) 
    
    # Define the final features matching the original model input
    feature_cols = [
        "temp", "humidity", "pressure", "wind_speed", "rain_1h", "rain_3h", "rain_6h", 
        "rain_24h", "rain_intensity", "temp_delta", "humidity_delta", 
        "hour", "day", "month", "year"
    ]
    
    X = df[feature_cols].fillna(0)
    y = df[TARGET_COLUMN]
    
    # Drop rows that don't have enough history for lag features (first few rows)
    Xy = pd.concat([X, y], axis=1).dropna()

    if Xy.empty or len(Xy) < 50:
         raise ValueError(f"Insufficient valid data ({len(Xy)} rows) for training.")

    # Time-based split: latest 20% for validation (avoids data leakage)
    split_idx = int(len(Xy) * 0.8)
    X_train, X_val = Xy.iloc[:split_idx][feature_cols], Xy.iloc[split_idx:][feature_cols]
    y_train, y_val = Xy.iloc[:split_idx][TARGET_COLUMN], Xy.iloc[split_idx:][TARGET_COLUMN]
    
    return X_train, X_val, y_train, y_val

# ------------------------------------------------------------------------------------------------

def run_retraining_pipeline(run_name):
    """
    Orchestrates the training, logs metrics to MLflow, and prepares the model for promotion.
    """
    mlflow.set_experiment("Flood_Risk_Retrain") # Separate experiment for governance
    
    try:
        df = pd.read_csv(RAW_LOG_PATH)
        X_train, X_val, y_train, y_val = preprocess_and_split(df)
        
    except ValueError as e:
        print(f"Skipping training due to: {e}")
        return {"f1": -2.0, "recall": -2.0, "run_id": "skipped"} # Use -2.0 to denote skip

    with mlflow.start_run(run_name=run_name) as run:
        print("Starting training run...")
        
        # --- Model Definition and Training (Keep original hyperparams) ---
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=2,
            min_samples_split=5, random_state=42, class_weight="balanced"
        )
        
        # Use Calibrated Classifier for robust probabilities
        model = CalibratedClassifierCV(rf, cv=3, method='isotonic')
        model.fit(X_train, y_train)
        
        # --- Validation ---
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        brier = brier_score_loss(y_val, y_proba)
        
        # --- MLflow Logging ---
        mlflow.log_params({"training_size": len(X_train), "validation_size": len(X_val)})
        mlflow.log_metric("validation_f1_score", f1)
        mlflow.log_metric("validation_recall_score", recall)
        mlflow.log_metric("validation_brier_loss", brier)
        
        # Log the model artifact and register it for promotion
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        
        # Push metrics to XCom for the next Airflow task (promotion)
        metrics = {"f1": f1, "recall": recall, "run_id": run.info.run_id}
        print(f"Training Complete. Metrics: {metrics}")
        return metrics

# ------------------------------------------------------------------------------------------------

def promote_model_callable(ti):
    """
    Compares the newly trained model against the current production model
    and promotes it if performance metrics meet the threshold.
    """
    new_model_metrics = ti.xcom_pull(task_ids='train_and_validate', key='return_value')
    if not new_model_metrics or new_model_metrics.get("f1", -2.0) == -2.0:
        print("No new model metrics or training was skipped. Skipping promotion.")
        return 'promotion_skipped'

    new_f1 = new_model_metrics['f1']
    new_run_id = new_model_metrics['run_id']
    
    client = mlflow.tracking.MlflowClient()

    # 1. Get the current Production Model metrics
    try:
        source_model = client.get_latest_versions(MODEL_NAME, stages=['Production'])[0]
        prod_model_run = client.get_run(source_model.run_id)
        prod_f1 = prod_model_run.data.metrics['validation_f1_score']
        print(f"Current Production F1: {prod_f1:.4f}")
    except (IndexError, mlflow.exceptions.RestException):
        # Handle case where no Production model exists (first run)
        prod_f1 = -1.0
        print("No existing Production model found. New model will be promoted.")

    print(f"New Model F1: {new_f1:.4f}. Minimum Threshold: {METRICS_THRESHOLD:.4f}")

    # 2. Promotion Logic
    if new_f1 > prod_f1 and new_f1 >= METRICS_THRESHOLD:
        
        # Find the version that corresponds to the new run ID
        for version in client.get_latest_versions(MODEL_NAME):
            if version.run_id == new_run_id:
                new_version = version.version
                break
        else:
             raise Exception(f"Could not find registered version for run ID {new_run_id}")

        # Transition the new model version to Production
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version,
            stage="Production",
            archive_existing_versions=True # Archive old production model
        )
        print(f"✅ Promoted Model Version {new_version} to Production (F1: {new_f1:.4f})")
        return 'deployment_successful'
    
    else:
        print(f"New model (F1: {new_f1:.4f}) did not beat Production model (F1: {prod_f1:.4f}) or meet the minimum threshold.")
        return 'promotion_skipped'