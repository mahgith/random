from typing import Dict, Any
from kfp import dsl
from kfp.dsl import Output, Dataset

# Docker image configuration
IMAGE = "europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images/base:latest"

@dsl.component(base_image=IMAGE)
def clean_and_merge_tables(
    infra: Dict[str, Any],
    job_params: Dict[str, Any],
    output_dataset: Output[Dataset]
) -> None:
    """
    Cleans Bronze data and merges shipments with their latest tracking events.

    This component executes a BigQuery SQL transformation to create the 'Enriched Shipments' 
    Silver table. It performs deduplication of tracking events using a window function 
    (ROW_NUMBER), cleans string-based weight fields using regex, casts data types for 
    analysis, and filters out rows with critical missing values.

    The component also outputs a KFP Dataset artifact containing metadata for downstream 
    traceability.

    Args:
        infra (Dict[str, Any]): 
            Global infrastructure configuration.
            - project_id (str): The GCP Project ID.
            - location (str): The GCP region for the BigQuery job.
        job_params (Dict[str, Any]): 
            Transformation or Business parameters.
            - bronze_dataset (str): Source dataset containing raw tables.
            - silver_dataset (str): Target dataset for the enriched table.
            - shipments_table (str): Name of the raw shipments table.
            - events_table (str): Name of the raw tracking events table.

        Outputs:
            output_dataset (Output[Dataset]): KFP Artifact that stores metadata 
                about the created Silver table, including its URI and Console link.              

    Returns:
        None.
    """

    import structlog
    from google.cloud import bigquery
    from common.core.logger import get_logger
    from common.core.helper import get_secret

    # Initialize the application structured logger
    logger = get_logger("cleaning-task") 

    try:

        # Unpack infrastructure and job configuration parameters into local variables
        project_id = infra["project_id"]
        location = infra["location"]
        bronze_dataset = job_params["bronze_dataset"]
        silver_dataset = job_params["silver_dataset"]
        shipments_table = job_params["shipments_table"]
        events_table = job_params["events_table"]

        # Construct the fully qualified table ID
        silver_table_id = f"{project_id}.{silver_dataset}.enriched_shipments"

        # Bind context variables. All subsequent logs will include these fields.
        # This is for filtering logs in GCP Cloud Logging.
        structlog.contextvars.bind_contextvars(
            project_id=project_id,
        )  

        # Initialize the BigQuery client for a specific project and region
        client = bigquery.Client(project=project_id, location=location)  

        # NOTE: The following logs are printed only as an example (demo purposes).
        # Do NOT log real secrets in production environments.
        db_password = get_secret(project_id,"DB_PASSWORD")
        logger.info("Getting secret.", db_password=db_password)
      

        # SQL logic for cleaning and merging
        # We use a CTE to deduplicate events (keeping only the most recent status)
        # and perform type casting for better downstream analysis.
        query = f"""
            CREATE OR REPLACE TABLE `{silver_table_id}` AS 
            WITH latest_event AS ( 
                SELECT * EXCEPT(row_num) 
                FROM ( 
                    SELECT *, 
                        ROW_NUMBER() OVER( 
                            PARTITION BY shipment_id 
                            ORDER BY timestamp DESC 
                        ) as row_num 
                    FROM `{project_id}.{bronze_dataset}.{events_table}`
                ) WHERE row_num = 1 
            ) 
            SELECT 
                s.shipment_id,
                s.origin_city,
                s.destination_city,
                s.carrier,
                s.weather_severity,
                s.traffic_density,
                SAFE_CAST(
                    REGEXP_REPLACE(s.weight_kg, r'[^0-9.]', '')
                AS FLOAT64) as weight_kg,
                s.distance_km,
                CAST(s.pickup_timestamp AS DATETIME) as pickup_timestamp,
                e.event_type as last_event_status,
                CAST(e.timestamp AS DATETIME) as last_event_timestamp,
                s.planned_duration_hrs,
                s.late_delivery 
            FROM `{project_id}.{bronze_dataset}.{shipments_table}` s 
            LEFT JOIN latest_event e 
            ON s.shipment_id = e.shipment_id
            WHERE s.origin_city IS NOT NULL 
                AND s.weight_kg IS NOT NULL 
                AND s.destination_city IS NOT NULL
        """        

        logger.info("Starting BigQuery cleaning and merge", 
                    shipments=shipments_table, 
                    events=events_table)
        
        query_job = client.query(query)
        query_job.result()

        # Saving metadata in output
        output_dataset.metadata["table_id"] = silver_table_id
        output_dataset.metadata["console_link"] = (
            f"https://console.cloud.google.com/bigquery?"
            f"project={project_id}&p={project_id}&d={silver_dataset}&t=enriched_shipments"
        )

        logger.info("Silver table created successfully", 
                    table=silver_table_id, 
                    rows=query_job.num_dml_affected_rows)

    except Exception as e:
        logger.error("Cleaning task failed", error_detail=str(e), exc_info=True)
        raise e