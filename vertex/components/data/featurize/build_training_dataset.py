from typing import Dict, Any
from kfp import dsl
from kfp.dsl import Input, Output, Dataset

# Docker image configuration
IMAGE = "europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images/base:latest"

@dsl.component(base_image=IMAGE)
def build_training_dataset(
    processed_data: Input[Dataset],
    training_dataset: Output[Dataset],
    infra: Dict[str, Any],
    job_params: Dict[str, Any],
) -> None:
    """
    Builds a feature‑engineered training dataset from processed input data and publishes it to the
    Gold layer in BigQuery, exposing its location via a KFP `Dataset` artifact for downstream tasks.

    This component loads the processed dataset referenced by the upstream KFP `Dataset` artifact,
    performs feature engineering (risk flags, temporal features, numeric coercions, and efficiency
    ratios), and writes the result to a managed BigQuery table in the Gold dataset. It records
    metadata (fully‑qualified table ID, feature count, and URI) in the output `Dataset` artifact
    for lineage and discoverability.

    Args:
        processed_data (Input[Dataset]):
            Upstream KFP dataset artifact providing:
            - table_id (str): Fully‑qualified BigQuery table with cleaned/processed input data.
        infra (Dict[str, Any]):
            Global infrastructure configuration.
            - project_id (str): GCP project where BigQuery jobs are executed.
            - location (str): GCP region for BigQuery operations.
        job_params (Dict[str, Any]):
            Transformation or Business parameters.
            - gold_dataset (str): Target BigQuery dataset (Gold layer) to store the training table.
            - risk_threshold (float|int): Threshold used to flag `is_high_risk_traffic`.
        
        Outputs:
            training_dataset (Output[Dataset]):
                Output artifact capturing:
                    - metadata.table_id (str): Fully‑qualified BigQuery table for the Gold training dataset.
                    - metadata.feature_count (int): Number of columns after feature engineering.
                    - uri (str): BigQuery URI of the output table (e.g., bq://project.dataset.table).

    Returns:
        None.

    """
    import structlog
    import pandas as pd
    import numpy as np    
    from google.cloud import bigquery
    from common.core.logger import get_logger    

    # Initialize the application structured logger
    logger = get_logger("feature-engineering-task") 

    try:

        # Unpack infrastructure and job configuration parameters into local variables
        project_id = infra["project_id"]
        location = infra["location"]
        gold_dataset = job_params["gold_dataset"]
        risk_threshold = job_params["risk_threshold"]

        # Obtain the table_id from the metadata of the previous component
        source_table = processed_data.metadata["table_id"]
        gold_table_id = f"{project_id}.{gold_dataset}.train_dataset_v1"

        # Initialize the BigQuery client for a specific project and region
        client = bigquery.Client(project=project_id, location=location)
       
        # Loading data with pandas
        logger.info("Loading data", table=source_table)
        df = client.query(f"SELECT * FROM `{source_table}`").to_dataframe()

        logger.info("Starting feature transformations", initial_rows=len(df))

        # Temporary variables
        df['is_high_risk_traffic'] = (df['traffic_density'] > risk_threshold).astype(int)  
        df['pickup_datetime'] = pd.to_datetime(df['pickup_timestamp'])
        df['hour'] = df['pickup_datetime'].dt.hour
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        # Convert weight_kg to a numeric in case it arrived as a string
        df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce').fillna(0)
        # Efficiency ratio (distance/planned time)
        df['planned_speed'] = df['distance_km'] / df['planned_duration_hrs']

        # Saving in BigQuery 
        logger.info("Uploading features to Gold layer")
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        client.load_table_from_dataframe(df, gold_table_id, job_config=job_config).result()

        # Saving metadata in output
        training_dataset.metadata["table_id"] = gold_table_id
        training_dataset.metadata["feature_count"] = len(df.columns)
        training_dataset.uri = f"bq://{gold_table_id}"

        logger.info("Gold features created successfully.", columns=list(df.columns))

    except Exception as e:
        logger.error("Feature engineering failed", error_detail=str(e), exc_info=True)
        raise e