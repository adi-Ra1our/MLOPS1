import json
import os

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "The_Cancer_data_1500_V2.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_data.csv")
VALIDATION_REPORT_PATH = os.path.join(PROCESSED_DATA_DIR, "validation_report.json")

EXPECTED_COLUMNS = [
    "Age",
    "Gender",
    "BMI",
    "Smoking",
    "GeneticRisk",
    "PhysicalActivity",
    "AlcoholIntake",
    "CancerHistory",
    "Diagnosis",
]

NUMERIC_COLUMNS = [
    "Age",
    "BMI",
    "Smoking",
    "GeneticRisk",
    "PhysicalActivity",
    "AlcoholIntake",
    "CancerHistory",
    "Diagnosis",
]


def validate_schema(df):
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if extra_columns:
        raise ValueError(f"Unexpected columns found: {extra_columns}")


def detect_outliers(df):
    outlier_summary = {}

    for col in NUMERIC_COLUMNS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_summary[col] = {
            "count": int(outlier_mask.sum()),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    return outlier_summary


def build_validation_report(df):
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "missing_values": {col: int(count) for col, count in df.isnull().sum().items()},
        "outliers": detect_outliers(df),
    }


def load_and_clean_data(path):
    df = pd.read_csv(path)

    print("Original shape:", df.shape)
    validate_schema(df)

    # Remove duplicates
    df = df.drop_duplicates()

    if "Gender" in df.columns:
        if df["Gender"].dtype == object:
            df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
        df["Gender"] = pd.to_numeric(df["Gender"], errors="coerce")

        if df["Gender"].isnull().any():
            mode_gender = df["Gender"].mode().iloc[0] if not df["Gender"].mode().empty else 0
            df["Gender"] = df["Gender"].fillna(mode_gender)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(df.mean(numeric_only=True))

    for col in df.columns:
        if df[col].isnull().any():
            mode = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode)

    print("After cleaning:", df.shape)

    return df


if __name__ == "__main__":
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    df = load_and_clean_data(RAW_DATA_PATH)
    validation_report = build_validation_report(df)

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    print("Data cleaned successfully")
    print(f"Validation report saved to: {VALIDATION_REPORT_PATH}")
