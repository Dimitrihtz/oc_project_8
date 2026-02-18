"""Generate synthetic production predictions with intentional data drift.

Writes 1000 prediction entries directly to logs/predictions.jsonl,
simulating 7 days of production predictions with drift on 3 features:
- EXT_SOURCE_2: mean shifted down by 0.15 (credit bureau change)
- DAYS_BIRTH: shifted +3000 toward younger applicants
- AMT_ANNUITY: increased 20% (inflation)
"""

import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "results" / "lightgbm_optimized.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "dataset_top10_features_data.csv"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "predictions.jsonl"

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


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def sample_with_drift(ref_data: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Sample a row from reference data and apply drift to 3 features."""
    row = ref_data.sample(n=1, random_state=int(rng.integers(0, 2**31))).iloc[0]
    features = row[FEATURE_COLUMNS].to_dict()

    # Drift: EXT_SOURCE_2 shifted down by 0.15
    features["EXT_SOURCE_2"] = max(0.0, min(1.0, features["EXT_SOURCE_2"] - 0.15))

    # Drift: DAYS_BIRTH shifted +3000 (younger applicants)
    features["DAYS_BIRTH"] = min(-1, int(features["DAYS_BIRTH"] + 3000))

    # Drift: AMT_ANNUITY increased 20%
    features["AMT_ANNUITY"] = round(features["AMT_ANNUITY"] * 1.20, 2)

    return features


def generate_timestamps(n: int, rng: np.random.Generator) -> list[datetime]:
    """Generate n sorted timestamps spread over DAYS_SPAN days."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=DAYS_SPAN)
    offsets = sorted(rng.uniform(0, DAYS_SPAN * 86400, size=n))
    return [start + timedelta(seconds=float(s)) for s in offsets]


def main():
    rng = np.random.default_rng(42)

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model()

    print(f"Loading reference data from {DATA_PATH}...")
    ref_data = pd.read_csv(DATA_PATH)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

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
            "timestamp": ts.isoformat(),
            "input_features": features,
            "prediction": prediction,
            "probability_default": round(probability, 6),
            "credit_decision": credit_decision,
        })

    with open(LOG_FILE, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    n_denied = sum(1 for e in entries if e["credit_decision"] == "denied")

    print(f"\nWritten {N_REQUESTS} entries to {LOG_FILE}")
    print(f"  Denied: {n_denied}/{N_REQUESTS} ({n_denied/N_REQUESTS:.1%})")


if __name__ == "__main__":
    main()
