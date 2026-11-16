# SentinelGraph

SentinelGraph — Network threat detection with graph neural networks.

A modular monorepo for ingesting network flow data, constructing graph
representations, training/evaluating GNN and classical ML threat-detection
models, and serving predictions through a FastAPI backend with a React +
TypeScript dashboard.

## Repository structure

```
SentinelGraph/
├── apps/                # Deployable applications
│   ├── api/             # FastAPI inference backend
│   └── dashboard/       # React + TypeScript frontend
├── ml/                  # Machine learning packages
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── classical_models/
│   ├── anomaly_detection/
│   ├── graph_learning/
│   ├── explainability/
│   └── evaluation/
├── data/                # Data stores (gitignored contents)
│   ├── raw/
│   ├── processed/
│   └── graphs/
├── notebooks/           # Exploratory analysis notebooks
├── artifacts/           # Build/training outputs (gitignored)
├── infrastructure/      # Docker, CI/CD, deployment configs
├── scripts/             # Helper and maintenance scripts
├── tests/               # Automated test suite
├── docs/                # Documentation
├── .github/             # GitHub workflows and templates
├── pyproject.toml       # Python packaging configuration
├── README.md
├── LICENSE
└── .gitignore
```

## Local development setup

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm/pnpm
- git

### Backend (Python)

From the repository root:

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install the package with dev dependencies
pip install -e ".[dev,api,ml]"

# Install pre-commit hooks (optional)
pre-commit install
```

Run the API:

```bash
cd apps/api
uvicorn sentinelgraph_api.main:app --reload
```

### Frontend (React + TypeScript)

From the repository root:

```bash
cd apps/dashboard
npm install
npm run dev
```

The dashboard is intentionally a minimal scaffold — no dashboard features are
implemented in this issue. See `apps/dashboard/README.md` for details.

### Running tests

```bash
# From the repository root
pytest
```

### Repository conventions

- Python code is formatted with `black` and `isort`, linted with `ruff`, and
  type-checked with `mypy`.
- Never commit datasets, model checkpoints, or weights (see `.gitignore`).
- Keep backend and frontend independently configurable.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution guide, code
standards, and release process.

## License

MIT License. See [LICENSE](LICENSE).
