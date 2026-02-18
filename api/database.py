import logging
import os

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import (
    Column,
    DateTime,
    Double,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    desc,
    func,
    insert,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("input_features", JSONB),
    Column("prediction", SmallInteger),
    Column("probability_default", Double),
    Column("credit_decision", String(10)),
)

reference_data = Table(
    "reference_data",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("TARGET", SmallInteger, nullable=False),
    Column("EXT_SOURCES_MEAN", Double, nullable=False),
    Column("CREDIT_TERM", Double, nullable=False),
    Column("EXT_SOURCE_3", Double, nullable=False),
    Column("GOODS_PRICE_CREDIT_PERCENT", Double, nullable=False),
    Column("INSTAL_AMT_PAYMENT_sum", Double, nullable=False),
    Column("AMT_ANNUITY", Double, nullable=False),
    Column("POS_CNT_INSTALMENT_FUTURE_mean", Double, nullable=False),
    Column("DAYS_BIRTH", Integer, nullable=False),
    Column("EXT_SOURCES_WEIGHTED", Double, nullable=False),
    Column("EXT_SOURCE_2", Double, nullable=False),
)

_engine: AsyncEngine | None = None


async def init_db():
    global _engine
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.info("DATABASE_URL not set — prediction logging will use JSONL")
        return

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)

    try:
        _engine = create_async_engine(database_url, pool_size=5, max_overflow=0)
        async with _engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
    except Exception:
        logger.exception("Failed to connect to PostgreSQL — falling back to JSONL")
        _engine = None
        return

    logger.info("PostgreSQL prediction logging initialized")


async def close_db():
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


def is_db_enabled() -> bool:
    return _engine is not None


async def insert_prediction(log_entry: dict):
    if _engine is None:
        return

    try:
        async with _engine.begin() as conn:
            await conn.execute(
                insert(predictions).values(
                    timestamp=log_entry["timestamp"],
                    input_features=log_entry["input_features"],
                    prediction=log_entry.get("prediction"),
                    probability_default=log_entry.get("probability_default"),
                    credit_decision=log_entry.get("credit_decision"),
                )
            )
    except Exception:
        logger.exception("Failed to insert prediction into PostgreSQL")


async def get_predictions(limit: int = 50, offset: int = 0) -> list[dict]:
    if _engine is None:
        return []

    async with _engine.connect() as conn:
        result = await conn.execute(
            select(predictions)
            .order_by(desc(predictions.c.timestamp))
            .limit(limit)
            .offset(offset)
        )
        return [row._asdict() for row in result]
