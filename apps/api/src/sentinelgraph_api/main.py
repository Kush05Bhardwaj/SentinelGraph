"""SentinelGraph API application factory."""

from __future__ import annotations

from fastapi import FastAPI

from sentinelgraph_api.core.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    return app


app = create_app()
