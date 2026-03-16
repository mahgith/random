"""
Loads pipeline_config.yaml and exposes settings as a simple object.
Import this in your components and pipeline files.

Usage:
    from configs.settings import settings
    print(settings.PROJECT_ID)
"""
import os
from pathlib import Path
import yaml

# Find the config file relative to this file
_CONFIG_PATH = Path(__file__).parent / "pipeline_config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load_config()


class Settings:
    # GCP
    PROJECT_ID: str = _cfg["gcp"]["project_id"]
    REGION: str = _cfg["gcp"]["region"]
    ARTIFACT_BUCKET: str = _cfg["gcp"]["artifact_bucket"]

    # Pipeline
    PIPELINE_NAME: str = _cfg["pipeline"]["name"]
    COMPILED_PIPELINE_PATH: str = _cfg["pipeline"]["compiled_pipeline_path"]

    # Data
    RAW_DATA_PATH: str = _cfg["data"]["raw_data_path"]
    PROCESSED_DATA_PATH: str = _cfg["data"]["processed_data_path"]
    LOOKBACK_DAYS: int = _cfg["data"]["lookback_days"]
    FORECAST_HORIZON: int = _cfg["data"]["forecast_horizon"]

    # Training
    MACHINE_TYPE: str = _cfg["training"]["machine_type"]
    BASE_IMAGE: str = _cfg["training"]["base_image"]
    HYPERPARAMETERS: dict = _cfg["training"]["hyperparameters"]

    # Model
    MODEL_DISPLAY_NAME: str = _cfg["model"]["display_name"]
    SERVING_CONTAINER: str = _cfg["model"]["serving_container"]

    # Experiment tracking
    EXPERIMENT_NAME: str = _cfg["experiment"]["name"]


settings = Settings()
