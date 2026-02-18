from datetime import datetime

from pydantic import BaseModel, Field


class CreditFeatures(BaseModel):
    EXT_SOURCES_MEAN: float = Field(
        ge=0.0, le=1.0,
        description="Mean of external credit scores",
        examples=[0.524],
    )
    CREDIT_TERM: float = Field(
        ge=0.0, le=1.0,
        description="Credit term ratio (annuity / credit amount)",
        examples=[0.05],
    )
    EXT_SOURCE_3: float = Field(
        ge=0.0, le=1.0,
        description="External source 3 score",
        examples=[0.535],
    )
    GOODS_PRICE_CREDIT_PERCENT: float = Field(
        ge=0.0, le=1.5,
        description="Goods price as percentage of credit amount",
        examples=[0.9],
    )
    INSTAL_AMT_PAYMENT_sum: float = Field(
        ge=0.0, le=1e8,
        description="Sum of installment payments",
        examples=[318619.5],
    )
    AMT_ANNUITY: float = Field(
        gt=0.0, le=1e6,
        description="Loan annuity amount",
        examples=[24903.0],
    )
    POS_CNT_INSTALMENT_FUTURE_mean: float = Field(
        ge=0.0, le=200.0,
        description="Mean count of future installments (POS)",
        examples=[6.95],
    )
    DAYS_BIRTH: int = Field(
        lt=0, ge=-30000,
        description="Client age in days (negative, relative to application date)",
        examples=[-15750],
    )
    EXT_SOURCES_WEIGHTED: float = Field(
        ge=0.0, le=3.0,
        description="Weighted combination of external sources",
        examples=[1.5],
    )
    EXT_SOURCE_2: float = Field(
        ge=0.0, le=1.0,
        description="External source 2 score",
        examples=[0.566],
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability_default: float
    credit_decision: str


class PredictionLog(BaseModel):
    id: int
    timestamp: datetime
    input_features: dict | None = None
    prediction: int | None = None
    probability_default: float | None = None
    credit_decision: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
