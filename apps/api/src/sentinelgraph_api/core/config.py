"""Application settings and configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application configuration loaded from environment variables."""

    app_name: str = Field(default="SentinelGraph API", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, description="Bind port")
    log_level: str = Field(default="info", description="Logging level")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
