# Contributing to SentinelGraph

Thank you for contributing to SentinelGraph! This guide covers the development
workflow, code standards, and release process.

## Getting started

1. Fork the repository and create a feature branch from `main`.
2. Set up the local development environment (see `README.md`).
3. Implement your changes, following the repository structure and conventions.
4. Add or update tests for any new functionality.
5. Run linting, type checks, and the test suite before opening a pull request.
6. Open a pull request against `main`.

## Code standards

### Python

- Format with `black` (line length 100).
- Sort imports with `isort` (black profile).
- Lint with `ruff`.
- Type-check with `mypy` (`disallow_untyped_defs = true`).
- All public functions and modules must include type hints and docstrings.

### TypeScript / React

- Format with Prettier.
- Lint with ESLint.
- Type-check with `tsc --noEmit`.
- Keep components small and typed; avoid `any` where possible.

## Running checks

```bash
# Python linting
ruff check apps/api ml scripts tests
ruff format --check apps/api ml scripts tests

# Python type checking
mypy apps/api ml scripts tests

# Python tests
pytest

# Frontend
cd apps/dashboard
npm run lint
npm run typecheck
npm run test
npm run build
```

## Branch naming

Use descriptive branch names such as:

- `feature/sg-001-repo-structure`
- `fix/api-health-endpoint`
- `docs/installation-guide`

## Commit messages

Use concise, imperative commit messages, optionally prefixed with the issue:

```
SG-001: Create repository structure
```

## Releases

Releases are managed through GitHub Releases and tagged in git. Do not commit
datasets, model checkpoints, weights, or credentials.

## Support

Open an issue for bugs, feature requests, or questions.
