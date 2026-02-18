# Credit Scoring MLOps

Binary credit default prediction system built with LightGBM, served through a FastAPI REST API with ONNX Runtime inference, and monitored via a Streamlit dashboard.

The model predicts whether a loan applicant is likely to default, returning a probability score and an **approved/denied** decision based on an optimized threshold of **0.10**.

## Architecture

```
┌──────────────┐      ┌──────────────────┐      ┌────────────┐
│   Streamlit  │─────▶│   FastAPI API     │─────▶│ PostgreSQL │
│  Dashboard   │      │  (ONNX Runtime)  │      │  (logs)    │
└──────────────┘      └──────────────────┘      └────────────┘
                              │
                      ┌───────┴────────┐
                      │ LightGBM model │
                      │   (.onnx)      │
                      └────────────────┘
```

- **API** — FastAPI app serving predictions via an ONNX-converted LightGBM model (462 estimators, 10 features)
- **Frontend** — Streamlit dashboard with interactive sliders, gauge chart, and prediction history
- **Database** — PostgreSQL for prediction logging (falls back to JSONL file when unavailable)
- **Monitoring** — Evidently-based data drift analysis
- **CI/CD** — GitHub Actions for testing and deployment via Docker Compose

## Features

| Feature | Description |
|---|---|
| `EXT_SOURCES_MEAN` | Mean of external credit scores (0–1) |
| `CREDIT_TERM` | Annuity / credit amount ratio (0–1) |
| `EXT_SOURCE_3` | External source 3 score (0–1) |
| `GOODS_PRICE_CREDIT_PERCENT` | Goods price as % of credit (0–1.5) |
| `INSTAL_AMT_PAYMENT_sum` | Sum of installment payments |
| `AMT_ANNUITY` | Loan annuity amount |
| `POS_CNT_INSTALMENT_FUTURE_mean` | Mean count of future POS installments |
| `DAYS_BIRTH` | Client age in days (negative) |
| `EXT_SOURCES_WEIGHTED` | Weighted external sources (0–3) |
| `EXT_SOURCE_2` | External source 2 score (0–1) |

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker & Docker Compose (for production deployment)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd project_8

# Install base dependencies
uv sync

# Install with API extras
uv sync --extra api

# Install all extras for development
uv sync --extra api --extra test --extra optimization --extra monitoring --extra frontend
```

## Configuration

Copy the example environment file and edit as needed:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | _(none — falls back to JSONL logging)_ |
| `API_URL` | API base URL (for Streamlit) | `http://localhost:8000` |

## Usage

### Run the API locally

```bash
uv run --extra api uvicorn api.app:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `/docs`.

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check (model loaded status) |
| `POST` | `/predict` | Get credit decision for an applicant |
| `GET` | `/predictions` | List prediction history (requires DB) |

#### Example prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "EXT_SOURCES_MEAN": 0.524,
    "CREDIT_TERM": 0.05,
    "EXT_SOURCE_3": 0.535,
    "GOODS_PRICE_CREDIT_PERCENT": 0.9,
    "INSTAL_AMT_PAYMENT_sum": 318619.5,
    "AMT_ANNUITY": 24903.0,
    "POS_CNT_INSTALMENT_FUTURE_mean": 6.95,
    "DAYS_BIRTH": -15750,
    "EXT_SOURCES_WEIGHTED": 1.5,
    "EXT_SOURCE_2": 0.566
  }'
```

Response:

```json
{
  "prediction": 0,
  "probability_default": 0.034271,
  "credit_decision": "approved"
}
```

### Run the Streamlit dashboard

```bash
uv run --extra frontend streamlit run streamlit_app.py
```

The dashboard provides:
- Interactive feature sliders to submit predictions
- Gauge chart visualizing default probability against the threshold
- Prediction history with approval rate metrics and distribution charts

### Run with Docker Compose (production)

```bash
docker compose up -d
```

This starts:
- **PostgreSQL 16** on port 5432
- **FastAPI** on port 8000 (with health checks)

## Testing

```bash
uv run --extra test --extra api pytest tests/ -v
```

Run with coverage:

```bash
uv run --extra test --extra api pytest tests/ -v --cov=api --cov-report=term-missing
```

The test suite covers:
- Health endpoint validation
- Model loading and shape verification
- Valid prediction responses (status, fields, ranges, decision logic)
- Input validation (missing fields, out-of-range values, wrong types)

## Monitoring

### Generate synthetic traffic with drift

```bash
uv run python monitoring/generate_traffic.py
```

Generates 1,000 synthetic predictions with intentional drift on 3 features (simulating 7 days of production data) and writes them to `logs/predictions.jsonl`.

### Seed the database

```bash
uv run --extra api python -m api.seed_db
```

Loads reference data and 1,000 drifted predictions into PostgreSQL.

### Drift analysis

Open `monitoring/drift_analysis.ipynb` to run Evidently data drift reports comparing reference data against production predictions.

## Project Structure

```
.
├── api/
│   ├── app.py              # FastAPI application and endpoints
│   ├── schemas.py           # Pydantic request/response models
│   ├── database.py          # Async PostgreSQL (SQLAlchemy) layer
│   ├── middleware.py         # Prediction logging middleware
│   └── seed_db.py           # Database seeding script
├── monitoring/
│   ├── generate_traffic.py  # Synthetic traffic generator with drift
│   └── drift_analysis.ipynb # Evidently drift analysis notebook
├── notebooks/
│   └── optimization_performance.ipynb  # ONNX optimization benchmarks
├── results/
│   ├── lightgbm_optimized.onnx  # Production model (ONNX format)
│   ├── lightgbm_optimized.pkl   # Original LightGBM model
│   └── ...                      # Threshold analysis, hyperparameters
├── tests/
│   └── test_api.py          # API test suite (22 tests)
├── streamlit_app.py         # Streamlit dashboard
├── train_model_improved_top_10.ipynb  # Model training notebook
├── docker-compose.yml       # Production stack (API + PostgreSQL)
├── Dockerfile               # Multi-stage build for the API
└── pyproject.toml           # Dependencies and project config
```

## CI/CD

Two GitHub Actions workflows:

- **test** — Runs on push/PR to `dev` and `main`. Installs dependencies with `uv`, runs pytest with coverage.
- **deploy** — Runs on push to `main`. Copies files to the server via SCP, then builds and starts containers with Docker Compose.
