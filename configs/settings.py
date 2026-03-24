"""
PROJECT SETTINGS + PARAMETER LOADER
=====================================
Thin wrapper around common.core.Settings that adds project-specific
fields and helpers to load per-direction pipeline parameters from
parameters/<direction>/params_v1.yaml.

Usage:
    from configs.settings import get_settings, build_data_prep_params, build_forecasting_params

    s = get_settings()
    data_prep_params  = build_data_prep_params("inbound")
    forecasting_params = build_forecasting_params("inbound")
"""

import json
from functools import lru_cache
from typing import Any, Dict

import yaml
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from common.core.settings import Settings as _BaseSettings


class ProjectSettings(_BaseSettings):
    """
    Extends the base Settings with project-level defaults.
    Values are read from env vars > .env > defaults (pydantic-settings priority).
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )

    PIPELINE_NAME: str = Field(default="package-volume-forecasting")
    COMPILED_DATA_PREP_PATH: str = Field(
        default="pipelines/compiled/data_prep_pipeline.json"
    )
    COMPILED_FORECASTING_PATH: str = Field(
        default="pipelines/compiled/forecasting_pipeline.json"
    )


@lru_cache(maxsize=1)
def get_settings() -> ProjectSettings:
    return ProjectSettings()


# Keep a module-level `settings` alias for backward compatibility
# with any code that does `from configs.settings import settings`
settings = get_settings()


# ── Per-direction parameter loader ────────────────────────────────────────────

def load_pipeline_params(direction: str, version: str = "params_v1") -> Dict[str, Any]:
    """
    Loads the parameter YAML for a given direction.

    Args:
        direction: "inbound" or "outbound"
        version:   YAML filename without extension (default "params_v1")

    Returns:
        Parsed YAML as a Python dict.
    """
    valid = ("inbound", "outbound")
    if direction not in valid:
        raise ValueError(f"direction must be one of {valid}, got '{direction}'")

    s = get_settings()
    yaml_path = s.BASE_DIR / "parameters" / direction / f"{version}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Parameter file not found: {yaml_path}\n"
            f"Expected location: parameters/{direction}/{version}.yaml"
        )

    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_data_prep_params(direction: str, version: str = "params_v1") -> dict:
    """
    Builds the parameter dict for data_prep_pipeline (Pipeline 1).
    Reads all values from parameters/<direction>/<version>.yaml.

    Args:
        direction: "inbound" or "outbound"
        version:   parameter file version (default "params_v1")

    Returns:
        Dict of typed values for aiplatform.PipelineJob(parameter_values=...).
    """
    cfg = load_pipeline_params(direction, version)

    infra = cfg.get("infra", {})

    return {
        # Identity
        "direction": direction,

        # GCP infra
        "project_id": infra["project_id"],
        "location":   infra["location"],

        # GCS → BQ raw load
        "gcs_uris_json":      json.dumps(cfg.get("gcs_uris", [])),
        "gcs_raw_schema_json": cfg.get("gcs_raw_schema", ""),
        "bq_raw_dataset":     cfg.get("bq_raw_dataset", "logistics"),
        "bq_raw_table":       cfg.get("bq_raw_table", f"package_scans_{direction}"),

        # BQ source tables for data_ingestion_op
        "bq_tables_json":  json.dumps(cfg.get("bq_tables", [])),
        "bq_columns_json": json.dumps(cfg.get("bq_columns", {})),

        # Intermediate raw series table (data_ingestion_op output)
        "bq_ingested_dataset": cfg.get("bq_ingested_dataset", "logistics"),
        "bq_ingested_table":   cfg.get("bq_ingested_table", f"raw_series_{direction}"),

        # Processed features table (preprocessing_op output — read by Pipeline 2)
        "bq_processed_dataset": cfg.get("bq_processed_dataset", "logistics"),
        "bq_processed_table":   cfg.get("bq_processed_table", f"processed_series_{direction}"),
    }


def build_forecasting_params(direction: str, version: str = "params_v1") -> dict:
    """
    Builds the parameter dict for forecasting_pipeline (Pipeline 2).
    Reads all values from parameters/<direction>/<version>.yaml.

    Args:
        direction: "inbound" or "outbound"
        version:   parameter file version (default "params_v1")

    Returns:
        Dict of typed values for aiplatform.PipelineJob(parameter_values=...).
    """
    cfg = load_pipeline_params(direction, version)

    infra  = cfg.get("infra", {})
    train  = cfg.get("params_training", {})
    evalu  = cfg.get("params_evaluation", {})
    champ  = cfg.get("params_champion_vs_challenger", {})
    reg    = cfg.get("params_register", {})

    return {
        # Identity
        "direction": direction,

        # GCP infra
        "project_id":      infra["project_id"],
        "region":          infra["region"],
        "location":        infra["location"],
        "artifact_bucket": infra["artifact_bucket"],

        # Processed data source (written by data_prep_pipeline)
        "bq_processed_dataset": cfg.get("bq_processed_dataset", "logistics"),
        "bq_processed_table":   cfg.get("bq_processed_table", f"processed_series_{direction}"),

        # Training
        "lookback_days":      train["lookback_days"],
        "half_life_days":     train["half_life_days"],
        "hp_grid_json":       json.dumps(train.get("hp_grid", {})),
        "tuning_num_cutoffs": int(train.get("tuning_num_cutoffs", 4)),
        "experiment_name":    train["experiment_name"],
        "run_prefix":         train.get("run_prefix", f"{direction}-tuning"),

        # Evaluation
        "forecast_horizon":      evalu["forecast_horizon"],
        "evaluation_start_date": evalu["evaluation_start_date"],
        "backtest_step_days":    evalu["backtest_step_days"],
        "mape_threshold":        evalu["max_allowed_mape"],
        "wape_threshold":        evalu.get("max_allowed_wape", evalu["max_allowed_mape"]),

        # Champion vs Challenger
        "model_display_name":  champ["model_display_name"],
        "champion_label_env":  champ.get("champion_label_env", "prod"),
        "champion_label_role": champ.get("champion_label_role", "champion"),
        "champ_delta_rel":     float(champ.get("delta_rel", 0.03)),
        "champ_delta_abs":     float(champ.get("delta_abs", 0.005)),

        # Registration
        "serving_container_image_uri": infra.get("serving_image", ""),
        "archived_label_role": reg.get("archived_label_role", "archived"),
    }
