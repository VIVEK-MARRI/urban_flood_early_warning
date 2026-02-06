import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

def test_single_class_failure():
    print("Testing Single Class Failure...")
    # Create valid feature columns
    feature_cols = [
        "temp", "humidity", "pressure", "wind_speed", "rain_1h", "rain_3h",
        "rain_6h", "rain_24h", "rain_intensity", "temp_delta", "humidity_delta",
        "hour", "day", "month", "year"
    ]
    
    # Create data with ONLY class 0
    X_train = pd.DataFrame(np.random.rand(100, len(feature_cols)), columns=feature_cols)
    y_train = pd.Series(np.zeros(100), name="flood_risk")
    
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    print(f"y_train unique values: {y_train.unique()}")
    
    try:
        # replicate the logic in retrain_logic.py
        model = CalibratedClassifierCV(rf_model, cv=3, method="isotonic")
        model.fit(X_train, y_train)
        print("❌ Training SUCCEEDED (Unexpected if we expected failure)")
    except Exception as e:
        print(f"✅ Training FAILED as expected with error: {e}")

def test_standard_failure():
    print("\nTesting Standard Data (Should Succeed)...")
    feature_cols = ["feat_1", "feat_2"]
    X_train = pd.DataFrame(np.random.rand(100, 2), columns=feature_cols)
    # mixed class
    y_train = pd.Series(np.random.randint(0, 2, 100))
    
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    try:
        model = CalibratedClassifierCV(rf_model, cv=3, method="isotonic")
        model.fit(X_train, y_train)
        print("✅ Training SUCCEEDED")
    except Exception as e:
        print(f"❌ Training FAILED (Unexpected) with error: {e}")

if __name__ == "__main__":
    test_single_class_failure()
    test_standard_failure()
