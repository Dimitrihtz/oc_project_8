import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from api.middleware import LOG_DIR, PredictionLoggingMiddleware
from api.schemas import CreditFeatures, HealthResponse, PredictionResponse

MODEL_PATH = Path("results/lightgbm_optimized.pkl")
OPTIMAL_THRESHOLD = 0.10

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    yield


app = FastAPI(
    title="Credit Scoring API",
    description="Binary credit default prediction using LightGBM",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(PredictionLoggingMiddleware)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CreditFeatures):
    df = pd.DataFrame([features.model_dump()])
    df = df[model.feature_name_]

    probability = model.predict_proba(df)[0, 1]
    prediction = int(probability >= OPTIMAL_THRESHOLD)
    credit_decision = "denied" if prediction == 1 else "approved"

    return PredictionResponse(
        prediction=prediction,
        probability_default=round(float(probability), 6),
        credit_decision=credit_decision,
    )
