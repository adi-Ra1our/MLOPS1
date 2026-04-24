import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from serve_api import app

def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["model_loaded"] is True
    assert resp.json()["scaler_loaded"] is True

def test_predict_endpoint_shape():
    client = TestClient(app)
    # Use the correct number of features for the trained model
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/processed/cleaned_data.csv'))
    # Drop Gender column if it is always 0 (not used in training)
    feature_cols = [c for c in df.columns if c != "Diagnosis"]
    # Check if Gender is all zeros or constant
    if "Gender" in feature_cols and df["Gender"].nunique() == 1:
        feature_cols.remove("Gender")
    features = df.loc[0, feature_cols].tolist()
    resp = client.post("/predict", json={"features": features})
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0 or np.isnan(data["probability"])

def test_predict_endpoint_rejects_wrong_feature_count():
    client = TestClient(app)
    resp = client.post("/predict", json={"features": [1.0, 2.0]})
    assert resp.status_code == 400
    assert "Expected" in resp.json()["detail"]
