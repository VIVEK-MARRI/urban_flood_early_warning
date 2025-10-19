# api/app.py
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import json

app = FastAPI(title="Urban Flood Early Warning System")

# Load trained model
model = joblib.load("models/optimized_rf_model.pkl")
with open("models/feature_columns.json") as f:
    feature_columns = json.load(f)


@app.get("/")
def root():
    return {"message": "Flood Early Warning API is running!"}

@app.post("/predict_flood")
def predict_flood(data: dict):
    try:
        input_df = pd.DataFrame([data])
        

        # Check for missing columns
        missing_cols = [col for col in feature_columns if col not in input_df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

        # Reorder columns
        input_df = input_df[feature_columns]

        # Predict
        prediction = model.predict(input_df)[0]

        # Handle probability safely
        if len(model.classes_) > 1:
            probability = model.predict_proba(input_df)[0][1]  # probability of class 1
        else:
            probability = model.predict_proba(input_df)[0][0]  # only class 0 exists

        return {
            "flood_risk": int(prediction),
            "flood_probability": round(float(probability), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
