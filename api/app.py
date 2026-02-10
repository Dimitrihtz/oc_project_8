from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException

from api.database import close_db, init_db
from api.middleware import LOG_DIR, PredictionLoggingMiddleware
from api.schemas import CreditFeatures, HealthResponse, PredictionResponse

ONNX_MODEL_PATH = Path("results/lightgbm_optimized.onnx")
OPTIMAL_THRESHOLD = 0.10

FEATURE_ORDER = [
    "EXT_SOURCES_MEAN",
    "CREDIT_TERM",
    "EXT_SOURCE_3",
    "GOODS_PRICE_CREDIT_PERCENT",
    "INSTAL_AMT_PAYMENT_sum",
    "AMT_ANNUITY",
    "POS_CNT_INSTALMENT_FUTURE_mean",
    "DAYS_BIRTH",
    "EXT_SOURCES_WEIGHTED",
    "EXT_SOURCE_2",
]

session = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(ONNX_MODEL_PATH))
    await init_db()
    yield
    await close_db()


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
        model_loaded=session is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CreditFeatures):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = features.model_dump()
    row = np.array([[data[f] for f in FEATURE_ORDER]], dtype=np.float32)

    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: row})
    probability = float(result[1][0][1])

    prediction = int(probability >= OPTIMAL_THRESHOLD)
    credit_decision = "denied" if prediction == 1 else "approved"

    return PredictionResponse(
        prediction=prediction,
        probability_default=round(probability, 6),
        credit_decision=credit_decision,
    )
