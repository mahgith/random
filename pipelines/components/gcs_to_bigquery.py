"""
GCS TO BIGQUERY COMPONENT
==========================
Loads one or more CSV files from Google Cloud Storage into a BigQuery table.

Use this component when raw data files land in GCS and are not yet available
as BigQuery tables. Run this pipeline first, then run the main forecasting
pipeline which reads from BigQuery.

Supports:
  - Single file:    gcs_uris_json='["gs://bucket/raw/data.csv"]'
  - Multiple files: gcs_uris_json='["gs://bucket/raw/jan.csv", "gs://bucket/raw/feb.csv"]'
  - Wildcard:       gcs_uris_json='["gs://bucket/raw/*.csv"]'

Schema:
  - Leave schema_json empty ("") to let BigQuery auto-detect column types from the CSV.
  - Provide an explicit schema to enforce types:
      schema_json='[{"name": "scan_date", "type": "DATE"}, {"name": "volume", "type": "INTEGER"}]'
  Auto-detect is convenient for exploration; explicit schema is safer for production.

Write disposition:
  - "WRITE_TRUNCATE"  — overwrite the table on every run (default, idempotent)
  - "WRITE_APPEND"    — append rows to an existing table
"""

from kfp.dsl import component, Output, Metrics

_FORECASTING_IMAGE = "europe-west1-docker.pkg.dev/your-gcp-project-id/ml-images/forecasting:latest"


@component(base_image=_FORECASTING_IMAGE)
def gcs_to_bigquery_op(
    project_id: str,
    gcs_uris_json: str,       # JSON list of GCS URIs, e.g. '["gs://bucket/raw/*.csv"]'
    bq_dataset: str,           # Target BQ dataset, e.g. "logistics"
    bq_table: str,             # Target BQ table,   e.g. "package_scans"
    location: str,             # GCP region matching your BQ dataset, e.g. "europe-west1"
    schema_json: str = "",     # JSON list of {name, type}; empty = auto-detect
    write_disposition: str = "WRITE_TRUNCATE",  # or "WRITE_APPEND"
    skip_leading_rows: int = 1,                 # 1 = skip CSV header row
    load_metrics: Output[Metrics] = None,       # type: ignore[assignment]
):
    """Load CSV files from GCS into a BigQuery table."""
    import json
    import structlog
    from google.cloud import bigquery
    from common.core.logger import get_logger

    logger = get_logger("gcs-to-bigquery")
    structlog.contextvars.bind_contextvars(
        project_id=project_id,
        bq_dataset=bq_dataset,
        bq_table=bq_table,
        location=location,
    )

    # ── Parse inputs ──────────────────────────────────────────────────────────
    uris = json.loads(gcs_uris_json)
    if not uris:
        raise ValueError("gcs_uris_json must contain at least one URI")

    full_table_id = f"{project_id}.{bq_dataset}.{bq_table}"

    structlog.contextvars.bind_contextvars(
        source_uris=uris,
        destination=full_table_id,
    )
    logger.info("Starting GCS → BigQuery load", num_uris=len(uris))

    # ── Build schema ──────────────────────────────────────────────────────────
    if schema_json.strip():
        schema_dicts = json.loads(schema_json)
        schema = [
            bigquery.SchemaField(f["name"], f["type"])
            for f in schema_dicts
        ]
        autodetect = False
        logger.info("Using explicit schema", num_fields=len(schema))
    else:
        schema = []
        autodetect = True
        logger.info("Using auto-detect schema")

    # ── Configure load job ────────────────────────────────────────────────────
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        autodetect=autodetect,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=skip_leading_rows,
        write_disposition=write_disposition,
    )

    # ── Run load job ──────────────────────────────────────────────────────────
    client = bigquery.Client(project=project_id, location=location)

    logger.info("Submitting load job to BigQuery")
    load_job = client.load_table_from_uri(
        uris,
        full_table_id,
        job_config=job_config,
    )

    load_job.result()  # blocks until complete

    # ── Report results ────────────────────────────────────────────────────────
    destination_table = client.get_table(full_table_id)
    rows_loaded = load_job.output_rows
    total_rows  = destination_table.num_rows

    logger.info(
        "Load job complete",
        job_id=load_job.job_id,
        rows_loaded=rows_loaded,
        total_rows_in_table=total_rows,
        write_disposition=write_disposition,
    )

    load_metrics.log_metric("rows_loaded", rows_loaded)
    load_metrics.log_metric("total_rows_in_table", total_rows)
    load_metrics.log_metric("num_source_uris", len(uris))
