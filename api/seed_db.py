"""Seed PostgreSQL with reference data and synthetic production predictions.

Phase 1: Load reference data from CSV into the reference_data table.
Phase 2: Generate 1000 synthetic predictions with drift into the predictions table.

Usage:
    uv run --extra api python -m api.seed_db
"""

import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "results" / "lightgbm_optimized.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "dataset_top10_features_data.csv"

DATABASE_URL = os.environ["DATABASE_URL"]

N_REQUESTS = 1000
DAYS_SPAN = 7
OPTIMAL_THRESHOLD = 0.10

FEATURE_COLUMNS = [
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


def sample_with_drift(ref_data: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Sample a row from reference data and apply drift to 3 features."""
    row = ref_data.sample(n=1, random_state=int(rng.integers(0, 2**31))).iloc[0]
    features = row[FEATURE_COLUMNS].to_dict()

    features["EXT_SOURCE_2"] = max(0.0, min(1.0, features["EXT_SOURCE_2"] - 0.15))
    features["DAYS_BIRTH"] = min(-1, int(features["DAYS_BIRTH"] + 3000))
    features["AMT_ANNUITY"] = round(features["AMT_ANNUITY"] * 1.20, 2)

    return features


def generate_timestamps(n: int, rng: np.random.Generator) -> list[datetime]:
    """Generate n sorted timestamps spread over DAYS_SPAN days."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=DAYS_SPAN)
    offsets = sorted(rng.uniform(0, DAYS_SPAN * 86400, size=n))
    return [start + timedelta(seconds=float(s)) for s in offsets]


def seed_reference_data(engine, ref_data: pd.DataFrame):
    """Truncate and re-insert reference data."""
    print(f"Seeding reference_data ({len(ref_data):,} rows)...")
    columns = ["TARGET"] + FEATURE_COLUMNS
    df = ref_data[columns].copy()

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE reference_data"))

    df.to_sql("reference_data", engine, if_exists="append", index=False, method="multi", chunksize=5000)
    print(f"  Inserted {len(df):,} rows into reference_data.")


def seed_predictions(engine, ref_data: pd.DataFrame):
    """Generate synthetic predictions with drift and insert into predictions table."""
    print(f"\nLoading model from {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    rng = np.random.default_rng(42)

    timestamps = generate_timestamps(N_REQUESTS, rng)
    entries = []

    print(f"Generating {N_REQUESTS} synthetic predictions...")
    for ts in timestamps:
        features = sample_with_drift(ref_data, rng)
        df = pd.DataFrame([features])
        df = df[model.feature_name_]

        probability = float(model.predict_proba(df)[0, 1])
        prediction = int(probability >= OPTIMAL_THRESHOLD)
        credit_decision = "denied" if prediction == 1 else "approved"

        entries.append({
            "timestamp": ts,
            "input_features": features,
            "prediction": prediction,
            "probability_default": round(probability, 6),
            "credit_decision": credit_decision,
        })

    from api.database import predictions as predictions_table

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE predictions"))
        for i in range(0, len(entries), 500):
            conn.execute(predictions_table.insert(), entries[i : i + 500])

    n_denied = sum(1 for e in entries if e["credit_decision"] == "denied")

    print(f"  Inserted {N_REQUESTS} entries into predictions.")
    print(f"  Denied: {n_denied}/{N_REQUESTS} ({n_denied / N_REQUESTS:.1%})")


def main():
    url = DATABASE_URL
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    print("Connecting to database...")
    engine = create_engine(url)

    # Create tables if they don't exist
    from api.database import metadata

    metadata.create_all(engine)

    ref_data = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(ref_data):,} rows from {DATA_PATH}")

    seed_reference_data(engine, ref_data)
    seed_predictions(engine, ref_data)

    engine.dispose()
    print("\nDone!")


if __name__ == "__main__":
    main()
