"""
DATA PREP PIPELINE  (Pipeline 1 — runs daily)
=============================================
Triggered when new CSV data arrives in GCS. Ingests, transforms, and writes
a fully-featured time series into BigQuery so Pipeline 2 (training) can read
it at any time without re-processing.

PIPELINE GRAPH
--------------
  [optional] gcs_to_bigquery   — load raw CSV(s) from GCS into BQ
                └─► data_ingestion   — query BQ, apply scan deduplication + weekend folding
                        └─► preprocessing   — add calendar, holiday, rolling features → BQ

USAGE
-----
    python scripts/run_pipeline.py --pipeline data_prep --direction inbound
    python scripts/run_pipeline.py --pipeline data_prep --direction outbound

OUTPUT
------
BigQuery table:  {project_id}.{bq_processed_dataset}.{bq_processed_table}
This table is the sole input to the training pipeline.
"""

from kfp import dsl
from kfp.dsl import pipeline

from pipelines.components.gcs_to_bigquery import gcs_to_bigquery_op
from pipelines.components.data_ingestion import data_ingestion_op
from pipelines.components.preprocessing import preprocessing_op


@pipeline(
    name="package-volume-data-prep",
    description="Daily data prep: GCS CSV → BigQuery raw → BigQuery processed features",
)
def data_prep_pipeline(
    # ── Identity ──────────────────────────────────────────────────────────────
    direction: str,                    # "inbound" or "outbound"

    # ── GCP ───────────────────────────────────────────────────────────────────
    project_id: str,
    location: str,                     # BQ region, e.g. "europe-west1"

    # ── GCS → BQ raw load (only needed when new CSV files arrive) ─────────────
    gcs_uris_json: str,                # JSON list of GCS URIs, e.g. '["gs://bucket/raw/*.csv"]'
    gcs_raw_schema_json: str = "",     # explicit schema or "" for auto-detect
    bq_raw_dataset: str = "logistics",
    bq_raw_table: str = "package_scans",

    # ── BQ source tables for data_ingestion_op ────────────────────────────────
    bq_tables_json: str = "",          # JSON list of {dataset, table, date_from?, date_to?}
    bq_columns_json: str = "",         # JSON dict mapping logical name → actual BQ column name

    # ── Intermediate BQ table (raw series written by data_ingestion_op) ───────
    bq_ingested_dataset: str = "logistics",
    bq_ingested_table: str = "raw_series",   # direction appended at pipeline level below

    # ── Output BQ table (processed features written by preprocessing_op) ──────
    bq_processed_dataset: str = "logistics",
    bq_processed_table: str = "processed_series",  # direction appended at pipeline level below
):
    # ── Step 1: Load CSV files from GCS into BigQuery ─────────────────────────
    load_task = gcs_to_bigquery_op(
        project_id=project_id,
        gcs_uris_json=gcs_uris_json,
        bq_dataset=bq_raw_dataset,
        bq_table=bq_raw_table,
        location=location,
        schema_json=gcs_raw_schema_json,
        write_disposition="WRITE_TRUNCATE",
    )
    load_task.set_display_name("Load — GCS CSV → BigQuery raw")
    load_task.set_caching_options(enable_caching=False)
    load_task.set_cpu_limit("2")
    load_task.set_memory_limit("4G")

    # ── Step 2: Ingest — deduplicate, fold weekends, produce daily (ds, y) ────
    ingest_task = data_ingestion_op(
        direction=direction,
        project_id=project_id,
        location=location,
        bq_tables_json=bq_tables_json,
        bq_columns_json=bq_columns_json,
        bq_output_dataset=bq_ingested_dataset,
        bq_output_table=bq_ingested_table,
    )
    ingest_task.set_display_name("Ingest — BQ deduplication + weekend folding")
    ingest_task.after(load_task)
    ingest_task.set_caching_options(enable_caching=False)
    ingest_task.set_cpu_limit("2")
    ingest_task.set_memory_limit("8G")

    # ── Step 3: Preprocessing — calendar, holiday, rolling features ────────────
    preprocess_task = preprocessing_op(
        direction=direction,
        project_id=project_id,
        location=location,
        bq_raw_dataset=bq_ingested_dataset,
        bq_raw_table=bq_ingested_table,
        bq_output_dataset=bq_processed_dataset,
        bq_output_table=bq_processed_table,
    )
    preprocess_task.set_display_name("Preprocessing — calendar + holiday features → BQ")
    preprocess_task.after(ingest_task)
    preprocess_task.set_caching_options(enable_caching=False)
    preprocess_task.set_cpu_limit("2")
    preprocess_task.set_memory_limit("8G")
