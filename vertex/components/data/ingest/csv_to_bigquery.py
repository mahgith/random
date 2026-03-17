from typing import Dict, Any
from kfp import dsl

# Docker image configuration
IMAGE = "europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images/base:latest"

@dsl.component(base_image=IMAGE)
def ingest_csv_to_bigquery(
    infra: Dict[str, Any],
    job_params: Dict[str, Any]
) -> None:
    """
    Loads a CSV file from Google Cloud Storage into a BigQuery table.

    This component performs an asynchronous load job from a GCS URI to a destination 
    BigQuery table. It dynamically builds the schema based on input parameters and 
    configures the job to overwrite existing data (WRITE_TRUNCATE). It also binds 
    context variables to a structured logger for enhanced traceability in Cloud Logging.

    Args:
        infra (Dict[str, Any]): 
            Global infrastructure configuration.
            - project_id (str): The GCP Project ID.
            - location (str): The GCP region for the job (e.g., 'europe-west9').
            - bucket_name (str): Source GCS bucket where the raw data resides.
        job_params (Dict[str, Any]): 
            Transformation or Business parameters.
            - dataset (str): The target BigQuery dataset ID.
            - table_id (str): The target BigQuery table ID.
            - file_name (str): The CSV file name located in the 'raw/' GCS prefix.
            - schema_fields (List[Dict[str, str]]): A list of dictionaries defining 
                the schema (e.g., [{"name": "col1", "type": "STRING"}]).

    Returns:
        None.
    """     
    
    import structlog
    from google.cloud import bigquery
    from common.core.logger import get_logger

    # Initialize the application structured logger
    logger = get_logger("ingestion-task") 
    
    try:
        # Unpack infrastructure and job configuration parameters into local variables
        project_id = infra["project_id"]
        location = infra["location"]
        bucket = infra["bucket_name"]
        dataset_id = job_params["dataset"]
        file_name = job_params["file_name"]
        table_id = job_params["table_id"]
        schema_fields = job_params["schema_fields"]

        # Construct the fully qualified table ID and the GCS source URI
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        gcs_uri = f"gs://{bucket}/raw/{file_name}"        

        # Bind context variables. All subsequent logs will include these fields.
        # This is for filtering logs in GCP Cloud Logging.
        structlog.contextvars.bind_contextvars(
            project_id=project_id,
            location=location,
            dataset=dataset_id,
            table_name=full_table_id,
            source_uri=gcs_uri
        )  

        # Create a SchemaField object (name, type) based on the list[dict]
        schema = [
            bigquery.SchemaField(f["name"], f["type"])
            for f in schema_fields
        ]

        # Initialize the BigQuery client for a specific project and region
        client = bigquery.Client(project=project_id, location=location)

        # Configure the Load Job settings
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,                # Skip the CSV header row
            write_disposition="WRITE_TRUNCATE"  # Overwrite table if it already exists
        )

        # Define the destination table reference (project.dataset.table)
        table_ref = f"{project_id}.{dataset_id}.{table_id}"

        logger.info("Submitting Load Job to BigQuery")
        # Trigger the asynchronous load job from the GCS URI
        load_job = client.load_table_from_uri(
            gcs_uri, 
            table_ref, 
            job_config=job_config
        )    

        # Wait for the job to complete successfully
        load_job.result()

        rows_inserted = load_job.output_rows
        logger.info("Ingestion completed successfully", 
                    job_id=load_job.job_id,
                    inserted_rows=rows_inserted)
        
    except Exception as e:
        logger.error("Ingestion failed", error_detail=str(e), exc_info=True)
        raise e