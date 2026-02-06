# airflow/dags/retrain_logic.py

import os
import json
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
import mlflow.tracking

# --- Configuration ---
RAW_LOG_PATH = "data/prediction_log.csv"
MODEL_NAME = "UrbanFloodClassifier"
TARGET_COLUMN = "flood_risk"
METRICS_THRESHOLD = 0.95  # Minimum F1 score required for promotion
TARGET_MODEL_FILE = "/opt/airflow/models/urban_flood_model.pkl"

# -----------------------------
# Feature Engineering & Labeling
# -----------------------------
def create_flood_label(df):
    """
    Recreates the flood risk proxy label.
    FIX: Removed shift() so label corresponds to CURRENT conditions.
    """
    label = (
        (df["rain_24h"] >= 1.8)
        & (df["humidity"] >= 75.0)
        & (df["pressure"] <= 1012.0)
    ).astype(int)

    return label


def preprocess_and_split(df):
    """Prepares features, derives labels, and splits data for training/validation."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df[TARGET_COLUMN] = create_flood_label(df)

    feature_cols = [
        "temp", "humidity", "pressure", "wind_speed", "rain_1h", "rain_3h",
        "rain_6h", "rain_24h", "rain_intensity", "temp_delta", "humidity_delta",
        "hour", "day", "month", "year"
    ]

    X = df[feature_cols].fillna(0)
    y = df[TARGET_COLUMN]
    Xy = pd.concat([X, y], axis=1).dropna()

    if Xy.empty or len(Xy) < 50:
        raise ValueError(f"Insufficient valid data ({len(Xy)} rows) for training.")

    # Check for class balance
    if Xy[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training data contains only one class. Need both Flood(1) and Safe(0) examples.")

    split_idx = int(len(Xy) * 0.8)
    X_train, X_val = Xy.iloc[:split_idx][feature_cols], Xy.iloc[split_idx:][feature_cols]
    y_train, y_val = Xy.iloc[:split_idx][TARGET_COLUMN], Xy.iloc[split_idx:][TARGET_COLUMN]

    return X_train, X_val, y_train, y_val


# -----------------------------
# Training & Logging
# -----------------------------
def run_retraining_pipeline(run_name):
    """Orchestrates the training, logs metrics to MLflow, and registers the model."""
    mlflow.set_experiment("Flood_Risk_Retrain")

    try:
        df = pd.read_csv(os.path.join("/opt/airflow", RAW_LOG_PATH))
        X_train, X_val, y_train, y_val = preprocess_and_split(df)
    except ValueError as e:
        print(f"Skipping training due to: {e}")
        return {"f1": -2.0, "recall": -2.0, "run_id": "skipped"}

    with mlflow.start_run(run_name=run_name) as run:
        print("Starting Random Forest training run...")

        # --- MODEL DEFINITION: Random Forest ---
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )

        model = CalibratedClassifierCV(rf_model, cv=3, method="isotonic")
        model.fit(X_train, y_train)

        # --- Validation ---
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        brier = brier_score_loss(y_val, y_proba)

        # --- MLflow Logging ---
        mlflow.log_metric("validation_f1_score", f1)
        mlflow.log_metric("validation_recall_score", recall)
        mlflow.log_metric("validation_brier_loss", brier)

        with open(os.path.join("/opt/airflow", "models/feature_columns.json"), "w") as f:
            json.dump(X_train.columns.tolist(), f)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        metrics = {"f1": f1, "recall": recall, "run_id": run.info.run_id}
        print(f"Training Complete. Metrics: {metrics}")
        return metrics


# -----------------------------
# Model Promotion Logic
# -----------------------------
def promote_model_callable(ti):
    """
    Promotes the new model if metrics meet the threshold AND deploys
    the model artifact to the shared volume for live pipeline use.
    """
    new_model_metrics = ti.xcom_pull(task_ids="train_and_validate", key="return_value")
    if not new_model_metrics or new_model_metrics.get("f1", -2.0) == -2.0:
        print("No new model metrics or training was skipped. Skipping promotion.")
        return "promotion_skipped"

    new_f1 = new_model_metrics["f1"]
    new_run_id = new_model_metrics["run_id"]

    client = mlflow.tracking.MlflowClient()

    # Get current Production model metrics
    try:
        source_model = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
        prod_model_run = client.get_run(source_model.run_id)
        prod_f1 = prod_model_run.data.metrics["validation_f1_score"]
        print(f"Current Production F1: {prod_f1:.4f}")
    except (IndexError, mlflow.exceptions.RestException):
        prod_f1 = -1.0
        print("No existing Production model found. New model will be promoted.")

    print(f"New Model F1: {new_f1:.4f}. Minimum Threshold: {METRICS_THRESHOLD:.4f}")

    # Promotion Decision
    if new_f1 > prod_f1 and new_f1 >= METRICS_THRESHOLD:
        versions = client.search_model_versions(f"run_id='{new_run_id}'")
        if not versions:
            raise Exception(f"Could not find registered version for run ID {new_run_id}")

        new_version = versions[0].version

        # Promote new model
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version,
            stage="Production",
            archive_existing_versions=True
        )

        # Deploy artifact locally for API/DAG access
        model_uri = f"models:/{MODEL_NAME}/{new_version}"
        local_model = mlflow.sklearn.load_model(model_uri)

        os.makedirs(os.path.dirname(TARGET_MODEL_FILE), exist_ok=True)
        joblib.dump(local_model, TARGET_MODEL_FILE)

        print(f"✅ Promoted Model Version {new_version} (RandomForest) to Production (F1: {new_f1:.4f})")
        print(f"✅ Deployed model artifact to {TARGET_MODEL_FILE} for immediate API/DAG use.")
        return "deployment_successful"

    else:
        print(
            f"New model (F1: {new_f1:.4f}) did not beat Production model (F1: {prod_f1:.4f}) "
            f"or meet the minimum threshold. Keeping old model."
        )
        return "promotion_skipped"
