"""
Loads pipeline_config.yaml and exposes settings.

Usage:
    from configs.settings import settings, get_direction_config
    print(settings.PROJECT_ID)
    dir_cfg = get_direction_config("inbound")
"""
import json
from pathlib import Path
import yaml

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

    # Shared
    FORECAST_HORIZON: int = _cfg["forecast_horizon"]
    EVALUATION_START_DATE: str = _cfg["evaluation_start_date"]
    BACKTEST_STEP_DAYS: int = _cfg["backtest_step_days"]
    EXPERIMENT_NAME: str = _cfg["experiment_name"]

    # Training infra
    MACHINE_TYPE: str = _cfg["training"]["machine_type"]
    BASE_IMAGE: str = _cfg["training"]["base_image"]


settings = Settings()


def get_direction_config(direction: str) -> dict:
    """Return the config dict for 'inbound' or 'outbound'."""
    valid = ("inbound", "outbound")
    if direction not in valid:
        raise ValueError(f"direction must be one of {valid}, got '{direction}'")
    return _cfg["directions"][direction]


def build_pipeline_params(direction: str) -> dict:
    """
    Build the parameter dict passed to the pipeline when submitting a run.
    All per-direction config is serialised here so components receive plain
    Python types and don't need to read the YAML file themselves.
    """
    dir_cfg = get_direction_config(direction)

    return {
        "direction": direction,
        "project_id": settings.PROJECT_ID,
        "region": settings.REGION,
        "artifact_bucket": settings.ARTIFACT_BUCKET,
        "experiment_name": settings.EXPERIMENT_NAME,
        # BQ sources — JSON string so KFP can pass it as a pipeline param
        "bq_tables_json": json.dumps(dir_cfg["bq_tables"]),
        "bq_columns_json": json.dumps(dir_cfg["bq_columns"]),
        # L1
        "lookback_days": dir_cfg["lookback_days"],
        "half_life_days": dir_cfg["half_life_days"],
        # L2B
        "prophet_changepoint_prior_scale": dir_cfg["prophet_changepoint_prior_scale"],
        # L3
        "lgbm_n_estimators": dir_cfg["lgbm_n_estimators"],
        "lgbm_learning_rate": dir_cfg["lgbm_learning_rate"],
        "lgbm_num_leaves": dir_cfg["lgbm_num_leaves"],
        # Evaluation
        "forecast_horizon": settings.FORECAST_HORIZON,
        "evaluation_start_date": settings.EVALUATION_START_DATE,
        "backtest_step_days": settings.BACKTEST_STEP_DAYS,
        "mape_threshold": dir_cfg["mape_threshold"],
        "model_display_name": dir_cfg["model_display_name"],
    }
