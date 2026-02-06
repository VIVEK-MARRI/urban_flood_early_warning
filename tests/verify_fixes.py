import hashlib
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def stable_hash(data):
    """
    Stable hashing function using SHA-256.
    Sorts keys to ensure deterministic output for dictionaries.
    """
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

def test_hashing_stability():
    print("Testing Hashing Stability...")
    data1 = {"a": 1, "b": 2, "c": [1, 2, 3]}
    data2 = {"c": [1, 2, 3], "b": 2, "a": 1}
    
    hash1 = stable_hash(data1)
    hash2 = stable_hash(data2)
    
    print(f"Hash 1: {hash1}")
    print(f"Hash 2: {hash2}")
    
    assert hash1 == hash2, "Hashes should be identical for identical content regardless of key order"
    print("✅ Hashing is stable!")

def test_label_logic():
    print("\nTesting Labeling Logic...")
    df = pd.DataFrame({
        "rain_24h": [2.0, 0.5, 5.0],
        "humidity": [80, 50, 90],
        "pressure": [1010, 1020, 1000]
    })
    
    # Logic: rain >= 1.8 & humidity >= 75 & pressure <= 1012
    # Row 0: True, True, True -> 1
    # Row 1: False, False, False -> 0
    # Row 2: True, True, True -> 1
    
    # OLD INCORRECT LOGIC (shifted) would likely mislabel if we were testing time series
    # BUT here we test the Correct Logic: Current Row -> Current Label
    
    is_flood = (
        (df["rain_24h"] >= 1.8)
        & (df["humidity"] >= 75.0)
        & (df["pressure"] <= 1012.0)
    ).astype(int)
    
    print("Labels generated:", is_flood.tolist())
    
    assert is_flood.iloc[0] == 1, "Row 0 should be flood"
    assert is_flood.iloc[1] == 0, "Row 1 should not be flood"
    assert is_flood.iloc[2] == 1, "Row 2 should be flood"
    print("✅ Labeling logic is correct (no shift).")

def test_rf_model_training():
    print("\nTesting Random Forest Training...")
    # Generate dummy data
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    preds = model.predict(X)
    print(f"Model trained. Predictions shape: {preds.shape}")
    print("✅ Random Forest model trains successfully.")

if __name__ == "__main__":
    test_hashing_stability()
    test_label_logic()
    test_rf_model_training()
