from typing import Dict, Any
from kfp import dsl
from kfp.dsl import Output, Dataset

# Docker image configuration
BASE = "europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images/base:latest"

@dsl.component(base_image=BASE)
def extract_data_from_bq(
    infra: Dict[str, Any],
    job_params: Dict[str, Any],
    dataset_output: Output[Dataset],
) -> None:
    """
    Extracts a dynamic 18-month historical dataset from a BigQuery table into a CSV artifact.

    This component connects to BigQuery and executes a query to retrieve the most 
    recent 18 months of data, calculated dynamically from the maximum `target_date` 
    present in the table. It standardizes the output column names to `date` and `volume` 
    for downstream model training compatibility (e.g., Prophet) and saves the resulting 
    DataFrame to a Kubeflow Pipeline dataset artifact. It also binds context variables 
    to a structured logger for enhanced traceability in Cloud Logging.

    Args:
        infra (Dict[str, Any]): 
            Global infrastructure configuration.
            - project_id (str): The GCP Project ID.
            - location (str): The GCP region for the query execution (e.g., 'europe-west9').
        job_params (Dict[str, Any]): 
            Extraction and business parameters.
            - dataset (str): The source BigQuery dataset ID.
            - table_id (str): The source BigQuery table ID containing the actual volume data.
        dataset_output (Output[Dataset]): 
            The KFP output artifact where the extracted CSV data will be written.

    Returns:
        None.
    """

    import structlog
    import pandas as pd
    from google.cloud import bigquery
    from common.core.logger import get_logger

    # Initialize the application structured logger
    logger = get_logger("bq-extractor-task")

    try:

        # Unpack infrastructure and job configuration parameters into local variables
        project_id = infra["project_id"]
        location = infra["location"]
        dataset_id = job_params["dataset"]
        table_id = job_params["table_id"]

        # Bind context variables. All subsequent logs will include these fields.
        # This is for filtering logs in GCP Cloud Logging.
        structlog.contextvars.bind_contextvars(
            project_id=project_id,
            location=location,
        )          

        # Initialize the BigQuery client for a specific project and region
        client = bigquery.Client(project=project_id, location=location)

        logger.info("Reading data from", dataset=dataset_id, table=table_id)

        query = f"""
            SELECT 
                target_date AS target_date, 
                volume AS volume
            FROM `{project_id}.{dataset_id}.{table_id}`
            WHERE target_date >= DATE_SUB(
                (SELECT MAX(target_date) FROM `{project_id}.{dataset_id}.{table_id}`), 
                INTERVAL 18 MONTH
            )
            ORDER BY target_date ASC
        """

        logger.info("Executing dynamic 18-month extraction query on BigQuery...")
        
        query_job = client.query(query)
        df = query_job.result().to_dataframe()

        if df.empty:
            logger.error("Expected columns 'target_date' and 'volume' not found. Skipping rename.")
            raise ValueError(f"No data found in {project_id}.{dataset_id}.{table_id}")

        if "target_date" in df.columns and "volume" in df.columns:
            df = df.rename(columns={"target_date": "ds", "volume": "y"})
        else:
            logger.error("Expected columns 'target_date' and 'volume' not found. Skipping rename.")
            raise ValueError("Expected columns 'target_date' and 'volume' not found. Skipping rename.")
            
        logger.info(f"Extraction complete. Fetched {len(df)} rows.")
        logger.info(f"Date range extracted: {df['ds'].min()} to {df['ds'].max()}")

        df.to_csv(dataset_output.path, index=False)
        logger.info("Data successfully saved to pipeline artifact.")

    except Exception as e:
        logger.error("Failure during BigQuery extraction", error=str(e))
        raise RuntimeError(f"Failure during BigQuery extraction: {e}") from e