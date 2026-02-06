import os
import json
import joblib
import pandas as pd
from datetime import datetime
import time
from functools import wraps

from fastapi import FastAPI, Request, Body
from starlette.responses import Response, JSONResponse
from starlette.concurrency import run_in_threadpool

# --- MLOps Dependencies ---
import mlflow
import mlflow.sklearn
from prometheus_client import Counter, Histogram, generate_latest

# --------------------------------------------------
# FastAPI App Configuration
# --------------------------------------------------
app = FastAPI(title="🌧️ Urban Flood Early Warning System")

# --------------------------------------------------
# Global Configurations
# --------------------------------------------------
MODEL_NAME = "UrbanFloodClassifier"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}/Production"
LOCAL_MODEL_PATH = "models/urban_flood_model.pkl"
FEATURES_PATH = "models/feature_columns.json"

# --------------------------------------------------
# Prometheus Metrics
# --------------------------------------------------
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
PREDICTION_COUNTER = Counter('flood_predictions_total', 'Total flood predictions', ['risk_level'])
PROBA_HISTOGRAM = Histogram('flood_probability_score', 'Flood probability score distribution',
                            buckets=[0.0, 0.2, 0.5, 0.75, 0.9, 1.0])

# --------------------------------------------------
# Decorator for Monitoring
# --------------------------------------------------
def monitor_endpoint(endpoint_name):
    """Decorator to monitor latency and count for each endpoint."""
    def wrapper(func):
        @wraps(func)
        async def decorator(request: Request, *args, **kwargs):
            start_time = time.time()
            method = request.method

            REQUEST_COUNT.labels(method=method, endpoint=endpoint_name).inc()
            try:
                response = await run_in_threadpool(func, request, *args, **kwargs)
            except Exception as e:
                # Instead of raising — return structured error response
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": str(e)}
                )
            finally:
                duration = time.time() - start_time
                REQUEST_LATENCY.labels(method=method, endpoint=endpoint_name).observe(duration)

            return response
        return decorator
    return wrapper

# --------------------------------------------------
# Load Model & Features
# --------------------------------------------------
model = None
feature_columns = []

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
        print("✅ Model loaded from MLflow registry.")
    except Exception as e:
        print(f"⚠️ MLflow load failed ({e}). Falling back to local model...")
        if os.path.exists(LOCAL_MODEL_PATH):
            model = joblib.load(LOCAL_MODEL_PATH)
            print("✅ Model loaded from local disk.")
        else:
            print(f"🚨 No model found at {LOCAL_MODEL_PATH}. Using dummy model instead.")
            class DummyModel:
                def predict(self, X): return [0]
                def predict_proba(self, X): return [[0.7, 0.3]]
            model = DummyModel()

    # Load or fallback feature columns
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            feature_columns = json.load(f)
        print(f"✅ Feature columns loaded ({len(feature_columns)} features).")
    else:
        feature_columns = [
            "temp", "humidity", "pressure", "wind_speed",
            "rain_1h", "rain_3h", "rain_6h", "rain_24h",
            "rain_intensity", "temp_delta", "humidity_delta",
            "hour", "day", "month", "year"
        ]
        print(f"⚠️ Feature file not found, using default list ({len(feature_columns)}).")

except Exception as e:
    print(f"🚨 Startup Error: {e}")
    model = None

# --------------------------------------------------
# API Routes
# --------------------------------------------------

@app.get("/")
@monitor_endpoint("/")
async def root(request: Request):
    """Root route to verify API health."""
    status = "✅ Ready" if model is not None else "⚠️ Model unavailable"
    return {"message": "Flood Early Warning API is running!", "model_status": status}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4")


@app.post("/predict_flood")
@monitor_endpoint("/predict_flood")
def predict_flood(request: Request, data: dict = Body(...)):
    """Main flood prediction endpoint."""
    try:
        if model is None:
            return {"status": "error", "message": "Model not loaded."}

        # Convert incoming data to DataFrame
        input_df = pd.DataFrame([data])
        now = datetime.now()

        # Add time-based features
        for col in ["hour", "day", "month", "year"]:
            if col not in input_df.columns:
                input_df[col] = getattr(now, col)

        # Fill missing columns
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0.0

        input_df = input_df[feature_columns]

        # Predict
        prediction = int(model.predict(input_df)[0])
        proba = float(model.predict_proba(input_df)[0][1]) if hasattr(model, "predict_proba") else float(prediction)

        PREDICTION_COUNTER.labels(risk_level=str(prediction)).inc()
        PROBA_HISTOGRAM.observe(proba)

        return {
            "status": "success",
            "flood_risk": prediction,
            "flood_probability": round(proba, 3)
        }

    except Exception as e:
        PREDICTION_COUNTER.labels(risk_level="error").inc()
        return {"status": "error", "message": f"Prediction failed: {str(e)}"}
