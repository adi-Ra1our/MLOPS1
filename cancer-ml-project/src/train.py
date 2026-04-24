import os
import tempfile
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as ml_sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
TMP_DIR = os.path.join(BASE_DIR, ".tmp")


os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# Keep temporary MLflow artifacts inside the project so fresh runs work in
# restricted environments and in CI.
tempfile.tempdir = TMP_DIR
os.environ["TMPDIR"] = TMP_DIR
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR

mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")
mlflow.set_experiment("cancer-binary-classification")


df = pd.read_csv(DATA_PATH)

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# Drop constant features so training matches inference expectations and avoids
# carrying useless columns through the pipeline.
constant_columns = [col for col in X.columns if X[col].nunique() <= 1]
if constant_columns:
    X = X.drop(columns=constant_columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle missing values before scaling and training.
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
}

results = []

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test)
            else:
                y_proba = y_pred
            roc_auc = roc_auc_score(y_test, y_proba)
        except Exception:
            roc_auc = float("nan")

        mlflow.log_param("model", name)
        mlflow.log_metrics(
            {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "roc_auc": float(roc_auc),
            }
        )
        try:
            ml_sklearn.log_model(model, "model")
        except (OSError, PermissionError) as exc:
            warnings.warn(
                f"Skipping MLflow model artifact logging for {name}: {exc}",
                RuntimeWarning,
            )

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "ROC_AUC": roc_auc,
            }
        )

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

print("\nModel Comparison:\n", results_df)

results_df.to_csv(os.path.join(ARTIFACTS_DIR, "results.csv"), index=False)

plt.figure()
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.xticks(rotation=30)
plt.title("Model Accuracy Comparison")
plt.savefig(os.path.join(ARTIFACTS_DIR, "results_accuracy.png"))

plt.figure()
plt.bar(results_df["Model"], results_df["F1"])
plt.xticks(rotation=30)
plt.title("Model F1 Score Comparison")
plt.savefig(os.path.join(ARTIFACTS_DIR, "results_f1.png"))

results_df = results_df.sort_values(by="F1", ascending=False)
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))

print(f"\nBest Model: {best_model_name}")
