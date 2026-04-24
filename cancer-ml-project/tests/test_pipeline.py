import os
import sys
import pytest
import pandas as pd

# Add src to path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def test_cleaned_data_exists():
    """Test that cleaned data file is created and not empty."""
    cleaned_path = os.path.join(os.path.dirname(__file__), '../data/processed/cleaned_data.csv')
    validation_path = os.path.join(os.path.dirname(__file__), '../data/processed/validation_report.json')
    assert os.path.exists(cleaned_path), "Cleaned data file does not exist."
    assert os.path.exists(validation_path), "Validation report does not exist."
    df = pd.read_csv(cleaned_path)
    assert not df.empty, "Cleaned data file is empty."
    assert 'Diagnosis' in df.columns, "Diagnosis column missing."
    assert df.isnull().sum().sum() == 0, "There are missing values in cleaned data."

def test_model_artifacts_exist():
    """Test that best model and scaler are saved after training."""
    model_path = os.path.join(os.path.dirname(__file__), '../models/best_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), '../models/scaler.pkl')
    assert os.path.exists(model_path), "Best model file does not exist."
    assert os.path.exists(scaler_path), "Scaler file does not exist."

def test_results_csv():
    """Test that results.csv exists and contains expected columns."""
    results_path = os.path.join(os.path.dirname(__file__), '../artifacts/results.csv')
    assert os.path.exists(results_path), "results.csv does not exist."
    df = pd.read_csv(results_path)
    for col in ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
        assert col in df.columns, f"{col} missing in results.csv"
    assert len(df) >= 1, "results.csv should have at least one row."
