# Deployment Architecture

This document describes how SentinelGraph is deployed and run across its four
architectural boundaries: source code (GitHub), training environment (Kaggle),
persistent artifact storage (`artifacts/`), and the independent inference
runtime (FastAPI + React + PostgreSQL).

## 1. Boundary overview

| Boundary | Technology | Network access | Responsibility |
|----------|-----------|----------------|----------------|
| **Source code** | GitHub repository | Optional (clone/install) | Version-controlled code: `apps/`, `ml/`, `scripts/`, `tests/`, `docs/`. |
| **Training environment** | Kaggle Kernels / local equivalent | Offline during training | Executes training scripts; writes artifacts to persistent storage. |
| **Persistent artifact storage** | Local `artifacts/` directory | N/A | Stores model weights, metrics, explanations (gitignored, never committed). |
| **Inference runtime** | FastAPI (`apps/api`) + React (`apps/dashboard`) + PostgreSQL | Offline | Serves predictions, incidents, explanations; dashboard UI. |

## 2. Deployment diagram

```mermaid
flowchart TD
    subgraph GH["Source code — GitHub"]
        CODE[Source code: apps/, ml/, scripts/, tests/, docs/]
    end

    subgraph KAGGLE["Training environment — Kaggle (offline)"]
        TRAIN[Training scripts]
        KAGGLE_DATA[Dataset mounts (read-only)]
    end

    subgraph PAS["Persistent artifact storage — artifacts/ (gitignored)"]
        MODELS[Model weights]
        METRICS[Evaluation metrics]
        EXPLANATIONS[Explanation artifacts]
        FEATURES[Feature artifacts]
    end

    subgraph INF["Inference runtime — independent application runtime"]
        API[FastAPI backend — apps/api]
        DASH[React dashboard — apps/dashboard]
        DB[(PostgreSQL — predictions, incidents, model metadata)]
    end

    GH -->|export/train on| KAGGLE
    KAGGLE -->|writes artifacts| PAS
    PAS -->|loads model at startup| API
    PAS -->|reads explanations| DASH
    API <-->|reads/writes| DB
    DASH <-->|HTTP/REST| API
```

## 3. Environment provisioning

### Backend (FastAPI)

```bash
cd apps/api
uvicorn sentinelgraph_api.main:app --reload
```

- Serves `/health`, `/predict`, `/incidents`, `/incidents/{id}`.
- Loads the active model from `artifacts/` at startup.
- Persists predictions and incidents to PostgreSQL.

### Frontend (React dashboard)

```bash
cd apps/dashboard
npm run dev
```

- Consumes the FastAPI API over HTTP.
- Renders attack distributions, incident lists, explanations.

### PostgreSQL

Run via Docker Compose (see `infrastructure/`). Stores:

- Network event metadata
- Prediction records
- Threat alerts / incidents
- Model version metadata
- Explanation references

## 4. Offline-first operation

After `pip install -e ".[dev,api,ml]"` and `npm install`, all runtime components
operate without external network access:

- Inference loads models from `artifacts/` (local disk).
- The dashboard calls the local FastAPI backend.
- PostgreSQL runs locally (Docker or system service).

No model is fetched from a remote model registry at inference time.

## 5. Data persistence locations

| Concern | Location | Git-tracked? |
|---------|----------|--------------|
| Source code | `apps/`, `ml/`, `scripts/`, `tests/` | Yes |
| Datasets (raw) | `data/raw/` | No (gitignored) |
| Processed features | `data/processed/` | No (gitignored) |
| Graphs | `data/graphs/` | No (gitignored) |
| Model artifacts | `artifacts/` | No (gitignored) |
| Runtime DB | PostgreSQL data volume | No (runtime data) |

## 6. Major architectural trade-offs (deployment)

| Trade-off | Decision | Rationale |
|-----------|----------|-----------|
| Training vs. inference runtime | **Separate** | Training (Kaggle) and inference (FastAPI) are decoupled; inference never trains, training never serves. |
| Artifact storage | **Local `artifacts/`** | Simplest offline-first option; avoids cloud storage dependencies and credentials. |
| Containerization | **Docker Compose (post-v1.0)** | v1.0 runs locally; Docker packaging is a later concern (SG-047). |
| Authentication | **None in v1.0** | v1.0 assumes infrastructure-level access control; auth is out of scope (see PRD exclusions). |

## Module responsibilities (deployment)

| Module | Responsibility |
|--------|----------------|
| `apps/api` | FastAPI inference backend; model loading; `/predict`, `/incidents` endpoints; PostgreSQL writes. |
| `apps/dashboard` | React frontend; API client; metric/incident/explanation views. |
| `ml/*` | Training-time code executed in Kaggle; produces artifacts consumed by `apps/api`. |
| `infrastructure/` | Docker Compose, environment variables, deployment configuration (post-v1.0). |
| `scripts/` | Helper scripts for training export, artifact promotion, and local dev. |
