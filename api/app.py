import json
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from functools import wraps
import time

from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response

# --- MLOPS DEPENDENCIES ---
from prometheus_client import Counter, Histogram, generate_latest
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
# -------------------------

app = FastAPI(title="Urban Flood Early Warning System")

# -----------------------------
# Configuration & Globals
# -----------------------------
MODEL_NAME = "UrbanFloodClassifier"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}/Production"
LOCAL_MODEL_PATH = "/app/models/optimized_rf_v2.pkl"
FEATURES_PATH = "/app/models/feature_columns.json"

# -----------------------------
# Prometheus Metrics
# -----------------------------
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
PREDICTION_COUNTER = Counter('flood_predictions_total', 'Total flood predictions', ['risk_level'])
PROBA_HISTOGRAM = Histogram(
    'flood_probability_score',
    'Flood probability score distribution',
    buckets=[0.0, 0.2, 0.5, 0.75, 0.9, 1.0]
)

# -----------------------------
# Endpoint Monitoring Decorator
# -----------------------------
def monitor_endpoint(endpoint_name):
    """Decorator to wrap FastAPI endpoints for monitoring. (FINAL STRUCTURAL FIX APPLIED)"""
    def wrapper(func):
        @wraps(func)
        async def decorator(request: Request, *args, **kwargs):
            start_time = time.time()
            method = request.method

            REQUEST_COUNT.labels(method=method, endpoint=endpoint_name).inc()

            try:
                # FIX: This safely runs the synchronous function in the threadpool
                response = await run_in_threadpool(func, request, *args, **kwargs)
            except Exception as e:
                REQUEST_LATENCY.labels(method=method, endpoint=endpoint_name).observe(time.time() - start_time)
                raise e

            REQUEST_LATENCY.labels(method=method, endpoint=endpoint_name).observe(time.time() - start_time)
            return response
        return decorator
    return wrapper

# -----------------------------
# Dynamic Model Initialization
# -----------------------------
model = None
explainer = None
feature_columns = []

try:
    # 1️⃣ Load MLflow Model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
    except Exception as registry_err:
        print(f"⚠️ MLflow Registry load failed: {registry_err}. Attempting local disk fallback...")
        if os.path.exists(LOCAL_MODEL_PATH):
            model = joblib.load(LOCAL_MODEL_PATH)
        else:
            raise FileNotFoundError(f"CRITICAL: No model found in MLflow OR local path {LOCAL_MODEL_PATH}.")
    print(f"✅ Model successfully loaded from source.")

    # 2️⃣ Load Feature Columns
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            feature_columns = json.load(f)
    else:
        feature_columns = [
            "temp", "humidity", "pressure", "wind_speed",
            "rain_1h", "rain_3h", "rain_6h", "rain_24h",
            "rain_intensity", "temp_delta", "humidity_delta",
            "hour", "day", "month", "year"
        ]
    print(f"✅ Feature columns loaded ({len(feature_columns)} features).")

except Exception as e:
    print(f"🚨 CRITICAL ERROR: Failed to start FastAPI due to model loading issue: {e}")
    model = None
    explainer = None
    raise RuntimeError(f"FastAPI startup failed — production model unavailable: {e}")

# -----------------------------
# API ROUTES
# -----------------------------
@app.get("/")
@monitor_endpoint("/")
async def root(request: Request):
    return {"message": "Flood Early Warning API is running!", "model_status": "Production"}

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4")

@app.post("/predict_flood")
@monitor_endpoint("/predict_flood")
def predict_flood(request: Request, data: dict):
    """Main Prediction Endpoint (Stabilized)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not loaded or ready.")

    try:
        input_df = pd.DataFrame([data])
        now = datetime.now()

        # --- Data preparation (Fill missing features) ---
        for col, value in [("hour", now.hour), ("day", now.day), ("month", now.month), ("year", now.year)]:
            if col in feature_columns and col not in input_df.columns:
                input_df[col] = value
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0.0

        input_df = input_df[feature_columns]

        # --- Prediction ---
        proba_output = model.predict_proba(input_df)
        # FIX: Use .item() for guaranteed scalar conversion
        proba = proba_output[0][list(model.classes_).index(1)].item()
        prediction = int(model.predict(input_df)[0].item())

        # --- SHAP Explainability (REMOVED) ---
        explanation = {}

        # --- Prometheus Metrics ---
        PREDICTION_COUNTER.labels(risk_level=str(prediction)).inc()
        PROBA_HISTOGRAM.observe(proba)

        return {
            "flood_risk": prediction,
            "flood_probability": round(float(proba), 3),
            "explanation": explanation
        }

    except Exception as e:
        PREDICTION_COUNTER.labels(risk_level='error').inc()
        print(f"❌ Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")
