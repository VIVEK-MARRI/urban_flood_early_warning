# api/app.py (UPDATED)

from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import json
from datetime import datetime
import requests
# --- NEW IMPORTS ---
from prometheus_client import Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response
from functools import wraps
import time

app = FastAPI(title="Urban Flood Early Warning System")

# -----------------------------
# Prometheus Metrics Configuration
# -----------------------------
REQUEST_COUNT = Counter(
    'http_requests_total', 'Total HTTP requests', ['method', 'endpoint']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint']
)
PREDICTION_COUNTER = Counter(
    'flood_predictions_total', 'Total flood predictions', ['risk_level']
)
# Histogram for tracking prediction probability distribution
PROBA_HISTOGRAM = Histogram(
    'flood_probability_score', 'Flood probability score distribution', buckets=[0.0, 0.2, 0.5, 0.75, 0.9, 1.0]
)


def monitor_endpoint(endpoint_name):
    """Decorator to wrap FastAPI endpoints for monitoring."""
    def wrapper(func):
        @wraps(func)
        async def decorator(request: Request, *args, **kwargs):
            start_time = time.time()
            method = request.method
            
            # Increment request counter
            REQUEST_COUNT.labels(method=method, endpoint=endpoint_name).inc()
            
            response = await func(request, *args, **kwargs)
            
            # Record latency
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint_name).observe(time.time() - start_time)
            return response
        return decorator
    return wrapper

# -----------------------------
# Load model and feature list
# -----------------------------
MODEL_PATH = "models/optimized_rf_v2.pkl"
FEATURES_PATH = "models/feature_columns.json"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

try:
    with open(FEATURES_PATH) as f:
        feature_columns = json.load(f)
    print(f"✅ Feature columns loaded ({len(feature_columns)} features).")
except Exception as e:
    raise RuntimeError(f"Failed to load feature columns: {e}")

# -----------------------------
# API Routes
# -----------------------------
@app.get("/")
@monitor_endpoint("/")
async def root(request: Request):
    # Change function signature to accept Request for monitoring decorator
    return {"message": "Flood Early Warning API is running!!!"}

# --- NEW METRICS ENDPOINT ---
@app.get("/metrics")
async def metrics():
    """Endpoint for Prometheus to scrape system and application metrics."""
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4")
# ----------------------------

@app.post("/predict_flood")
@monitor_endpoint("/predict_flood")
def predict_flood(data: dict):
    try:
        # ... (Keep all data loading, preparation, and feature engineering logic as is) ...
        input_df = pd.DataFrame([data])

        # Remove unwanted keys if somehow present
        for col in ['minute', 'second']:
            if col in input_df.columns:
                input_df.drop(columns=[col], inplace=True)

        now = datetime.now()

        # Add only required time columns (if not present)
        for col in ['hour', 'day', 'month', 'year']:
            if col not in input_df.columns:
                input_df[col] = getattr(now, col)
        
        # Add minute/second if the model expects it, ensuring they are populated
        if 'minute' in feature_columns and 'minute' not in input_df.columns:
            input_df['minute'] = now.minute
        if 'second' in feature_columns and 'second' not in input_df.columns:
            input_df['second'] = now.second


        # Add missing model features as 0
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Keep only model features
        input_df = input_df[feature_columns]

        # Predict
        prediction = model.predict(input_df)[0]
        proba = (
            model.predict_proba(input_df)[0][1]
            if hasattr(model, "predict_proba")
            else float(prediction)
        )
        
        # --- NEW METRICS LOGGING ---
        PREDICTION_COUNTER.labels(risk_level=str(int(prediction))).inc()
        PROBA_HISTOGRAM.observe(proba)
        # ---------------------------

        return {
            "flood_risk": int(prediction),
            "flood_probability": round(float(proba), 2)
        }

    except Exception as e:
        # Log error prediction for monitoring purposes
        PREDICTION_COUNTER.labels(risk_level='error').inc()
        raise HTTPException(status_code=500, detail=str(e))