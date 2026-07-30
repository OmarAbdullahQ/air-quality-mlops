# Names:
Omar Abdullah-Badr Al Zahrani-Emad-Omar 


# Riyadh Air Quality Intelligence Platform

> End-to-end MLOps system that predicts whether the next hour in Riyadh will
> have high PM2.5 pollution risk — from data collection to production deployment.

---

## 1. Project Overview

### Goal

Build a compact, production-style machine learning system that:

1. Collects hourly air quality and weather data from the Open-Meteo API.
2. Engineers time-aware features with leakage prevention.
3. Trains and compares classification models tracked in MLflow.
4. Serves real-time predictions through a FastAPI backend.
5. Provides an interactive Streamlit frontend for end users.
6. Monitors data drift with Evidently reports.
7. Runs the full stack in Docker Compose and deploys via Dokploy.

### Prediction Target

**`high_pollution_next_hour`** — binary label indicating whether the PM2.5
concentration in the next hour will exceed the team-defined threshold.

### Technology Stack

| Layer            | Technology                                   |
| ---------------- | -------------------------------------------- |
| Package manager  | [UV](https://docs.astral.sh/uv/)             |
| Language         | Python 3.13                                  |
| ML framework     | scikit-learn                                  |
| Experiment tracking | MLflow                                    |
| API backend      | FastAPI + Uvicorn                             |
| Frontend         | Streamlit                                     |
| Monitoring       | Evidently (DataDrift + DataSummary presets)    |
| Containerization | Docker, Docker Compose                        |
| Deployment       | Dokploy                                       |
| Linting          | Ruff                                          |
| Testing          | pytest                                        |

---

## 2. Project Architecture & Data Flow

### Repository Structure

```
air-quality-mlops/
├── app/
│   ├── api/main.py              # FastAPI prediction service
│   └── frontend/app.py          # Streamlit user interface
├── src/air_quality/
│   ├── collect.py               # Data collection from Open-Meteo
│   ├── features.py              # Feature engineering pipeline
│   ├── train.py                 # Model training & MLflow logging
│   └── monitoring.py            # Evidently drift report generation
├── tests/
│   ├── test_api.py              # API endpoint tests
│   └── test_features.py         # Feature pipeline tests
├── data/
│   ├── raw/                     # Immutable raw JSON from API
│   ├── processed/               # model_table.parquet
│   └── monitoring/              # reference.csv + predictions.csv
├── models/                      # Saved model artifact (model.joblib)
├── reports/                     # monitoring.html (Evidently report)
├── Dockerfile.api               # API container image
├── Dockerfile.frontend          # Frontend container image
├── docker-compose.yml           # Full stack orchestration
├── pyproject.toml               # Project metadata & dependencies
├── uv.lock                      # Locked dependency versions
├── .env.example                 # Environment variable template
├── .gitignore
└── report.md                    # This document
```

### Data Flow Diagram

```
Open-Meteo API (air quality + weather)
      │
      ▼
Data Collector (collect.py)
      │  saves immutable payload
      ▼
data/raw/air_quality.json
      │
      ▼
Feature Pipeline (features.py)
      │  lag features, rolling means, chronological target
      ▼
data/processed/model_table.parquet
      │
      ▼
Training Script (train.py)
      │  chronological split → train/valid/test
      ├──▶ MLflow runs (metrics + model artifacts)
      └──▶ models/model.joblib  +  data/monitoring/reference.csv
            │
            ▼
    FastAPI Service ◄──────── Streamlit Frontend
      │                        (HTTP calls only,
      │                         never loads model)
      ├──▶ prediction logs → data/monitoring/predictions.csv
      │
      ▼
Evidently Report (monitoring.py)
      │  compares reference vs. current
      ▼
reports/monitoring.html

All services → Docker Compose → Dokploy
```

### Key Design Decisions

- **Leakage prevention**: Lag features use `shift(1)` (past only). The target
  `high_pollution_next_hour` is created with `shift(-1)` but is never used as
  a feature. Rolling means use `shift(1).rolling(6)` to avoid including the
  current hour.
- **Chronological split**: 70% train / 15% validation / 15% test — preserving
  temporal order (no random shuffle).
- **Frontend separation**: Streamlit calls FastAPI over HTTP; it never imports
  or loads `model.joblib` directly.
- **Model loaded once**: The model artifact is loaded at module level when
  FastAPI starts — not per request.

---

## 3. Setup & Execution Instructions

### Prerequisites

- Python 3.13+
- [UV](https://docs.astral.sh/uv/) package manager
- Docker & Docker Compose (for containerized execution)

### Local Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/OmarAbdullahQ/air-quality-mlops.git
cd air-quality-mlops

# 2. Install all dependencies
uv sync

# 3. Collect data (requires internet)
uv run python -m air_quality.collect

# 4. Build features
uv run python -m air_quality.features

# 5. Start MLflow server (Terminal 1)
uv run mlflow server --host 0.0.0.0 --port 5000

# 6. Train models (Terminal 2)
uv run python -m air_quality.train

# 7. Start FastAPI (Terminal 2, after training)
uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

# 8. Start Streamlit (Terminal 3)
uv run streamlit run app/frontend/app.py --server.port 8501

# 9. Generate monitoring report (after making some predictions)
uv run python -m air_quality.monitoring
```

### Service URLs (Local)

| Service    | URL                          |
| ---------- | ---------------------------- |
| FastAPI    | http://localhost:8000        |
| Swagger UI | http://localhost:8000/docs   |
| Streamlit  | http://localhost:8501        |
| MLflow     | http://localhost:5000        |

### Docker Compose Execution

```bash
# Validate configuration
docker compose config

# Build and start all services
docker compose up --build -d

# Check service status
docker compose ps

# Smoke test the API
curl http://localhost:8000/health
curl http://localhost:8000/model-info

# View logs
docker compose logs -f api

# Stop services (keep volumes)
docker compose down

# Full reset (delete volumes — instructor only)
docker compose down -v
```

**Docker services:**

| Service    | Image / Build          | Port  | Notes                          |
| ---------- | ---------------------- | ----- | ------------------------------ |
| `api`      | Dockerfile.api         | 8000  | Health check, read-only model volume |
| `frontend` | Dockerfile.frontend    | 8501  | Depends on api (service_healthy)     |
| `mlflow`   | ghcr.io/mlflow/mlflow  | 5000  | SQLite backend, persistent volume    |

**Volumes:**

- `monitoring_data` — persists prediction logs at `/app/data/monitoring`
- `mlflow_data` — persists MLflow database and artifacts at `/mlflow`

### Dokploy Deployment Runbook

1. Push the repository to GitHub.
2. Create a new Docker Compose project in Dokploy and connect the repository
   (branch: `docker-mlops`).
3. Set the Compose file path to `docker-compose.yml`.
4. Add required environment variables in Dokploy (do not commit `.env`).
5. Assign a domain to Streamlit (port 8501) and FastAPI (port 8000).
6. Deploy, inspect build logs, and wait for the API health check to pass.
7. Verify `/health` and `/docs` on the API domain.
8. Submit a prediction from the Streamlit domain.
9. Restart the stack and verify that MLflow data and monitoring logs persist
   through the named volumes.

---

## 4. Training & Evaluation Summary

### Data

- **Source**: Open-Meteo Air Quality API + Archive Weather API
- **Location**: Riyadh, Saudi Arabia (24.7136°N, 46.6753°E)
- **Period**: 90 days of hourly observations
- **Variables**: PM2.5, PM10, temperature, relative humidity, wind speed

### Feature Set

| Feature                | Description                              |
| ---------------------- | ---------------------------------------- |
| `pm2_5`                | Current PM2.5 concentration              |
| `pm10`                 | Current PM10 concentration               |
| `temperature_2m`       | Temperature at 2m                        |
| `relative_humidity_2m` | Relative humidity at 2m                  |
| `wind_speed_10m`       | Wind speed at 10m                        |
| `hour`                 | Hour of day (0–23)                       |
| `day_of_week`          | Day of week (0–6)                        |
| `pm2_5_lag_1`          | PM2.5 one hour ago                       |
| `pm2_5_lag_3`          | PM2.5 three hours ago                    |
| `pm2_5_rolling_mean_6` | 6-hour rolling mean of PM2.5 (shifted)   |

### Models Trained

| Model                | Preprocessing                         | Hyperparameters                                |
| -------------------- | ------------------------------------- | ---------------------------------------------- |
| Logistic Regression  | Median imputation → Standard scaling  | `max_iter=1000`, `class_weight="balanced"`     |
| Random Forest        | Median imputation → Standard scaling  | `n_estimators=250`, `max_depth=10`, `class_weight="balanced"`, `random_state=42` |

### Evaluation Metrics

All models are evaluated on the **validation set** for selection, and the best
model is re-evaluated on the held-out **test set** for final reporting.

| Metric              | Description                                            |
| ------------------- | ------------------------------------------------------ |
| **F1-score**        | Primary metric — harmonic mean of precision and recall |
| **Recall**          | Sensitivity for the high-risk class                    |
| **ROC-AUC**         | Area under the ROC curve                               |

### Decision Threshold

- **Threshold**: `0.50` (selected on validation data only)
- The threshold is saved inside the model artifact (`models/model.joblib`) and
  used consistently by the FastAPI service at inference time.

### MLflow Experiment Tracking

- **Experiment name**: `riyadh-air-quality`
- **Runs logged**: One per candidate model (logistic regression, random forest)
- **Logged per run**: model name, threshold, validation F1, validation recall,
  validation ROC-AUC, and the full sklearn pipeline artifact

---

## 5. Quality Assurance & Testing

### Running Tests

```bash
# Run all tests
uv run pytest -q

# Run with coverage
uv run pytest --cov=app --cov=src -q
```

### Test Suite

| Test File            | Test                              | Validates                                      |
| -------------------- | --------------------------------- | ---------------------------------------------- |
| `test_api.py`        | `test_health`                     | GET `/health` → 200, `model_loaded=true`       |
| `test_api.py`        | `test_invalid_prediction`         | POST `/predict` with incomplete JSON → 422     |
| `test_features.py`   | Feature pipeline tests            | Dataset shape, columns, no leakage             |

### Linting

```bash
# Check code style
uv run ruff check .

# Auto-fix violations
uv run ruff check . --fix
```

### Acceptance Tests (from guide)

| Test                  | Command                                              | Pass Condition                                         |
| --------------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| Environment           | `uv sync && uv run pytest -q`                        | Deps install and tests pass                            |
| Data pipeline         | `uv run python -m air_quality.collect && uv run python -m air_quality.features` | Raw and processed files created                        |
| Training              | `uv run python -m air_quality.train`                 | Model artifact created, metrics printed                |
| API health            | `GET /health`                                        | HTTP 200 and `model_loaded=true`                       |
| Input validation      | `POST /predict` with incomplete JSON                 | HTTP 422                                               |
| Valid prediction      | `POST /predict` with complete JSON                   | Probability between 0 and 1                            |
| Frontend separation   | Stop API service                                     | Streamlit reports "API unavailable"                    |
| Containers            | `docker compose up --build -d`                       | All services healthy/reachable                         |
| Monitoring            | Open `reports/monitoring.html`                       | Reference and current batches compared                 |
| Deployment            | Open Dokploy domains                                 | Frontend and API accessible externally                 |

### Smoke Test Commands

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info

# Valid prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pm2_5": 25.0,
    "pm10": 60.0,
    "temperature_2m": 32.0,
    "relative_humidity_2m": 25.0,
    "wind_speed_10m": 12.0,
    "hour": 14,
    "day_of_week": 3,
    "pm2_5_lag_1": 24.0,
    "pm2_5_lag_3": 22.0,
    "pm2_5_rolling_mean_6": 23.0
  }'

# Invalid prediction (should return 422)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pm2_5": 10}'
```

---

## Environment Variables

| Variable   | Default                  | Description               |
| ---------- | ------------------------ | ------------------------- |
| `API_URL`  | `http://localhost:8000`  | FastAPI base URL used by Streamlit |

---

## Scope

The following items are **intentionally out of scope** per project guidelines:
PostgreSQL, authentication, scheduled orchestration, Kubernetes,
Prometheus/Grafana, SHAP, CI/CD, and automated retraining.
