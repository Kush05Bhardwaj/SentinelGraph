# SentinelGraph development setup

This document describes how to set up SentinelGraph locally for development.

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- git

## Backend setup

```bash
# From the repository root
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev,api,ml]"
pre-commit install   # optional
```

Run the API:

```bash
cd apps/api
uvicorn sentinelgraph_api.main:app --reload
```

Visit `http://localhost:8000/health` to confirm the backend is running.

## Frontend setup

```bash
cd apps/dashboard
npm install
npm run dev
```

Visit `http://localhost:5173` to view the dashboard scaffold.

## Running tests

```bash
# From the repository root
pytest
```

## Linting and type checks

```bash
# Python
ruff check apps/api ml scripts tests
ruff format --check apps/api ml scripts tests
mypy apps/api ml scripts tests

# Frontend
cd apps/dashboard
npm run lint
npm run typecheck
```

## Important notes

- Never commit datasets, model checkpoints, weights, or credentials.
- Data under `data/` and outputs under `artifacts/` are gitignored.
- Keep the backend and frontend independently configurable.
