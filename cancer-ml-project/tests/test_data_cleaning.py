import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_cleaning import (
    EXPECTED_COLUMNS,
    build_validation_report,
    load_and_clean_data,
    validate_schema,
)

def test_load_and_clean_data():
    raw_path = os.path.join(os.path.dirname(__file__), '../data/raw/The_Cancer_data_1500_V2.csv')
    df = load_and_clean_data(raw_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'Diagnosis' in df.columns
    assert df.isnull().sum().sum() == 0, "There are missing values after cleaning."
    # Check Gender encoding if present
    if 'Gender' in df.columns:
        assert set(df['Gender'].unique()).issubset({0, 1}), "Gender not encoded as 0/1."

def test_validate_schema_rejects_missing_column():
    df = pd.DataFrame(columns=[col for col in EXPECTED_COLUMNS if col != "Diagnosis"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)

def test_build_validation_report_contains_outlier_summary():
    raw_path = os.path.join(os.path.dirname(__file__), '../data/raw/The_Cancer_data_1500_V2.csv')
    df = load_and_clean_data(raw_path)
    report = build_validation_report(df)

    assert report["row_count"] > 0
    assert report["column_count"] == len(df.columns)
    assert "outliers" in report
    assert "BMI" in report["outliers"]
    assert "count" in report["outliers"]["BMI"]
