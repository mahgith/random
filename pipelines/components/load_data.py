"""
LOAD DATA COMPONENT
===================
Reads the processed feature table from BigQuery (written by preprocessing_op
in data_prep_pipeline) and materialises it as a KFP Dataset artifact so that
all subsequent training-pipeline components can consume it without hitting
BigQuery individually.

This is the single BQ read point for the entire forecasting pipeline.
All downstream components (training, evaluation, champion_vs_challenger, refit)
receive the data via Input[Dataset] artifacts.
"""

from kfp.dsl import component, Output, Dataset

_FORECASTING_IMAGE = "europe-west1-docker.pkg.dev/your-gcp-project-id/ml-images/forecasting:latest"


@component(base_image=_FORECASTING_IMAGE)
def load_data_op(
    project_id: str,
    location: str,
    bq_processed_dataset: str,
    bq_processed_table: str,
    processed_data: Output[Dataset],
):
    """Read the processed feature table from BigQuery and write it as a Dataset artifact."""
    import pandas as pd
    import structlog
    from google.cloud import bigquery
    from common.core.logger import get_logger

    logger = get_logger("load-data")
    structlog.contextvars.bind_contextvars(
        project_id=project_id,
        source=f"{project_id}.{bq_processed_dataset}.{bq_processed_table}",
    )

    client = bigquery.Client(project=project_id, location=location)
    table  = f"`{project_id}.{bq_processed_dataset}.{bq_processed_table}`"

    df = client.query(f"SELECT * FROM {table} ORDER BY ds").to_dataframe()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    logger.info(
        "Loaded from BigQuery",
        rows=len(df),
        columns=list(df.columns),
        date_min=str(df["ds"].min().date()),
        date_max=str(df["ds"].max().date()),
    )

    df.to_parquet(processed_data.path + ".parquet", index=False)
    logger.info("Written to artifact", path=processed_data.path)
