import os
from enum import Enum
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Settings(BaseSettings):
    """
    Handles application settings: Env Vars > .env file > Defaults.
    """    
   
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore"
    )

    # Calculated project root path based on local files structure
    BASE_DIR: Path = Path(__file__).resolve().parents[3]

    # Logging configuration
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO)

    # Cloud Detection
    IS_VERTEX_AI: bool = Field(default=False)

    # By default is False, so it displays correctly on local
    LOG_JSON_FORMAT: bool = Field(default=False)


    @model_validator(mode="after")
    def auto_configure_environment(self):
        """
        Post-load logic.
        Configures defaults based on the detected environment.
        """
        # Cloud Detection
        # Detect if we are in the cloud by looking for a variable that Google injects automatically
        if os.getenv("CLOUD_ML_JOB_ID") or os.getenv("K_SERVICE"):
            self.IS_VERTEX_AI = True
            # Force JSON in Vertex AI for Cloud Logging indexing, Console for local dev
            self.LOG_JSON_FORMAT = True
            # Use current working directory as root when running in Vertex AI
            self.BASE_DIR = Path("/app")                  
           
        return self

@lru_cache
def get_settings() -> Settings:
    return Settings()