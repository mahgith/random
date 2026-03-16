"""
DATA INGESTION COMPONENT
========================
Responsibility: Pull raw data from its source and write it to GCS
                so downstream components can read it.

What goes here (from your notebook):
  - Your data loading cells (BigQuery query / GCS read / API call / etc.)
  - Any very basic schema validation (check expected columns exist)
  - Nothing else — no feature engineering, no cleaning

Output: A GCS path pointing to the raw data file (CSV or Parquet).
        Downstream components receive this path as input.

HOW KFP OUTPUTS WORK
---------------------
KFP components communicate by writing values to Output objects.
  - Simple values (str, int, float): use `Output[str]` etc.
  - Files/datasets: use `Output[Dataset]` — KFP manages the GCS URI for you.
  - You write to `output.path`, KFP handles uploading to GCS.
"""

from kfp.v2.dsl import component, Output, Dataset
from kfp.v2 import dsl


# ── Component definition ──────────────────────────────────────────────────────
# @component turns a regular Python function into a pipeline step.
# packages_to_install: pip packages available inside this component's container.
# base_image: the Docker image this step runs in.
#             Using a pre-built Vertex AI image keeps things simple at first.
@component(
    packages_to_install=[
        "pandas",
        "pyarrow",          # for parquet support
        "google-cloud-bigquery",
        "google-cloud-storage",
        "db-dtypes",        # needed for BQ → pandas type conversions
    ],
    base_image="python:3.10-slim",
)
def data_ingestion_op(
    # ── Inputs ────────────────────────────────────────────────────────────────
    # These become the parameters you pass when you call this component in the
    # pipeline. They are plain Python types — KFP serialises them.
    project_id: str,
    raw_data_gcs_path: str,     # GCS URI where we will write the output
    lookback_days: int = 365,

    # ── Outputs ───────────────────────────────────────────────────────────────
    # Output[Dataset] tells KFP "this component produces a file artifact".
    # KFP gives you a local staging path at `raw_dataset.path`.
    # You write your file there; KFP uploads it to GCS automatically.
    raw_dataset: Output[Dataset] = None,  # type: ignore[assignment]
):
    """Ingest raw data and write to GCS as parquet."""
    import pandas as pd
    import logging
    from pathlib import Path
    from datetime import datetime, timedelta

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # ── OPTION A: Load from BigQuery ──────────────────────────────────────────
    # Uncomment and adapt this block if your data lives in BigQuery.
    # Replace the query with your actual SQL.
    #
    # from google.cloud import bigquery
    # client = bigquery.Client(project=project_id)
    # cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    # query = f"""
    #     SELECT
    #         date,
    #         store_id,
    #         product_id,
    #         sales_quantity
    #     FROM `{project_id}.your_dataset.your_table`
    #     WHERE date >= '{cutoff}'
    #     ORDER BY date
    # """
    # log.info(f"Running BQ query (lookback={lookback_days} days) ...")
    # df = client.query(query).to_dataframe()

    # ── OPTION B: Load from GCS ───────────────────────────────────────────────
    # If your raw data is already in GCS as CSV/parquet, read it directly.
    # Uncomment and adapt.
    #
    # from google.cloud import storage
    # import io
    # gcs_source = "gs://your-bucket/raw/data.csv"
    # log.info(f"Loading data from GCS: {gcs_source}")
    # df = pd.read_csv(gcs_source)

    # ── PLACEHOLDER — replace with Option A or B above ───────────────────────
    # This synthetic data lets you run the pipeline end-to-end while you
    # work on plugging in your real data source.
    import numpy as np
    log.info("Using synthetic placeholder data — replace with your real source!")
    dates = pd.date_range(end=datetime.utcnow(), periods=lookback_days, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "value": np.random.randn(lookback_days).cumsum() + 100,  # fake time series
    })
    # ── END PLACEHOLDER ───────────────────────────────────────────────────────

    log.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # ── Validate basic schema ─────────────────────────────────────────────────
    # Add your expected columns here so you catch data issues early.
    required_columns = ["date", "value"]  # adjust to match your actual schema
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Data is missing expected columns: {missing}")

    # ── Write output ──────────────────────────────────────────────────────────
    # raw_dataset.path is a local temp path. Write your file here.
    # KFP will upload it to GCS and pass the GCS URI to downstream components.
    output_path = raw_dataset.path + ".parquet"
    df.to_parquet(output_path, index=False)
    log.info(f"Wrote {len(df)} rows to {output_path}")
