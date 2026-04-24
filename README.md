# Cancer Prediction ML Project

## Overview
This project is an end-to-end machine learning workflow for binary cancer
prediction. It covers data cleaning, model training, experiment tracking,
artifact generation, reproducible pipeline execution, testing, and API-based
serving.

The project compares multiple classification models and saves the best one for
inference through a FastAPI service.

## Project Features
- Data cleaning and preprocessing
- Multi-model training and evaluation
- Best-model selection and artifact saving
- Experiment tracking with MLflow
- Reproducible pipeline execution with DVC
- REST API serving with FastAPI
- Automated tests with pytest
- CI workflow for preprocessing, training, and tests
- Docker-based deployment setup

## Models Compared
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM

## Current Best Model
Based on the latest training run in [artifacts/results.csv](./artifacts/results.csv),
the best saved model is `Random Forest` with:

- Accuracy: `0.84`
- Precision: `0.7982`
- Recall: `0.7845`
- F1 Score: `0.7913`
- ROC-AUC: `0.9160`

## Dataset Schema
The cleaned dataset contains these columns:

- `Age`
- `Gender`
- `BMI`
- `Smoking`
- `GeneticRisk`
- `PhysicalActivity`
- `AlcoholIntake`
- `CancerHistory`
- `Diagnosis`

`Diagnosis` is the target column.

## Pipeline Flow
The pipeline works like this:

1. Raw data is read from `data/raw/The_Cancer_data_1500_V2.csv`.
2. `src/data_cleaning.py` removes duplicates, encodes fields, and fills missing values.
3. Cleaned data is saved to `data/processed/cleaned_data.csv`.
4. `src/train.py` trains and evaluates multiple models.
5. The best model and scaler are saved to `models/`.
6. Evaluation results are saved to `artifacts/`.
7. `src/serve_api.py` loads the saved model and exposes prediction endpoints.

## Project Structure
```text
cancer-ml-project/
|-- data/
|   |-- raw/
|   `-- processed/
|-- artifacts/
|-- models/
|-- src/
|   |-- data_cleaning.py
|   |-- train.py
|   `-- serve_api.py
|-- tests/
|-- dvc.yaml
|-- requirements.txt
|-- Dockerfile
`-- README.md
```

## Setup
Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The Pipeline
Run preprocessing:

```bash
python src/data_cleaning.py
```

Run training:

```bash
python src/train.py
```

Or run the DVC pipeline:

```bash
dvc repro
```

## Run Tests
```bash
pytest
```

## Run The API
Start the FastAPI server:

```bash
python -m uvicorn src.serve_api:app --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

## Prediction API
Endpoint:

```text
POST /predict
```

Request format:

```json
{
  "features": [58.0, 16.0853, 0.0, 1.0, 8.1463, 4.1482, 1.0]
}
```

Current feature order used by the trained model:

1. `Age`
2. `BMI`
3. `Smoking`
4. `GeneticRisk`
5. `PhysicalActivity`
6. `AlcoholIntake`
7. `CancerHistory`

Note:
`Gender` exists in the cleaned dataset, but it is currently constant in the
data and is dropped during training, so it is not expected by the saved model.

Example response:

```json
{
  "prediction": 1,
  "probability": 0.84
}
```

## Experiment Tracking
MLflow is used to log:

- model name
- accuracy
- precision
- recall
- F1 score
- ROC-AUC

Tracking artifacts are stored locally in `mlruns/`.

## Reproducibility
DVC tracks the preprocessing and training stages defined in
[dvc.yaml](./dvc.yaml).

## CI
A GitHub Actions workflow is included to:

- install dependencies
- run preprocessing
- run training
- run tests

Workflow file:
`.github/workflows/ci.yml`

## Docker Deployment
Build the image:

```bash
docker build -t cancer-api .
```

Run the container:

```bash
docker run -p 8000:8000 cancer-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Outputs
After training, the project generates:

- `models/best_model.pkl`
- `models/scaler.pkl`
- `artifacts/results.csv`
- `artifacts/results_accuracy.png`
- `artifacts/results_f1.png`

## Next Planned Improvements
- Monitoring and API logging
- Data validation and outlier detection
- Remote model registry integration
- Automated retraining and batch inference
