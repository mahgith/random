"""
APPLICATION SETTINGS
====================
Pydantic-based settings that auto-detect whether code is running locally
or inside Vertex AI and adjust defaults accordingly.

Priority order (highest → lowest):
    Environment variables  →  .env file  →  Defaults

Usage (inside a KFP component or the run script):
    from common.core.settings import get_settings
    s = get_settings()
    print(s.IS_VERTEX_AI, s.LOG_LEVEL)
"""

import os
from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Application-level settings shared across all components and scripts.
    Env vars > .env file > defaults.
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )

    # Project root — resolved from this file's location (3 levels up: core → common → root)
    BASE_DIR: Path = Path(__file__).resolve().parents[2]

    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO)

    # Populated automatically by the validator below
    IS_VERTEX_AI: bool = Field(default=False)

    # False by default so local dev shows coloured console output;
    # set to True automatically when running inside Vertex AI.
    LOG_JSON_FORMAT: bool = Field(default=False)

    @model_validator(mode="after")
    def auto_configure_environment(self) -> "Settings":
        """
        Detects the execution environment and sets cloud-specific defaults.
        Google injects CLOUD_ML_JOB_ID for Vertex AI Training / Pipelines.
        K_SERVICE is present in Cloud Run environments.
        """
        if os.getenv("CLOUD_ML_JOB_ID") or os.getenv("K_SERVICE"):
            self.IS_VERTEX_AI = True
            self.LOG_JSON_FORMAT = True
            # Inside the container the project is always mounted at /app
            self.BASE_DIR = Path("/app")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
