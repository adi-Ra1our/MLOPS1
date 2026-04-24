import logging
import os
import time
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("cancer_api")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

scaler = joblib.load(SCALER_PATH)
best_model = joblib.load(MODEL_PATH)

app = FastAPI(title="Cancer Prediction API")


class PredictRequest(BaseModel):
    features: List[float]


class PredictResponse(BaseModel):
    prediction: int
    probability: float


EXPECTED_FEATURES = getattr(scaler, "n_features_in_", None)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info("Request started: method=%s path=%s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.exception(
            "Request failed: method=%s path=%s duration_ms=%s",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(
        "Request completed: method=%s path=%s status_code=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    feature_count = len(req.features)
    logger.info("Prediction requested: feature_count=%s", feature_count)

    if EXPECTED_FEATURES is not None and feature_count != EXPECTED_FEATURES:
        logger.warning(
            "Prediction rejected: expected_features=%s received_features=%s",
            EXPECTED_FEATURES,
            feature_count,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} features but received {feature_count}.",
        )

    try:
        X = np.array(req.features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = int(best_model.predict(X_scaled)[0])

        if hasattr(best_model, "predict_proba"):
            prob = float(best_model.predict_proba(X_scaled)[0, 1])
        else:
            prob = float("nan")

        logger.info("Prediction completed: prediction=%s probability=%s", pred, prob)
        return PredictResponse(prediction=pred, probability=prob)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": best_model is not None,
        "scaler_loaded": scaler is not None,
        "expected_features": EXPECTED_FEATURES,
    }
