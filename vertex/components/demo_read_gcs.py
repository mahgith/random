"""
Step 1 — Read a CSV from GCS.

Connects to GCS, downloads the file, parses it as a DataFrame,
and saves it as a Dataset artifact for the next step.

No transformation is applied here.
"""

from kfp.dsl import component, Output, Dataset

BASE_IMAGE = "europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images/base:latest"


@component(base_image=BASE_IMAGE)
def demo_read_gcs_op(
    project_id: str,
    gcs_uri: str,           # e.g. "gs://my-bucket/demo/data.csv"
    raw_data: Output[Dataset],
):
    import pandas as pd
    from common.core.logger import get_logger

    logger = get_logger("demo-read-gcs")

    logger.info("Reading CSV from GCS", uri=gcs_uri)

    # gcsfs is installed in the base image — pd.read_csv handles gs:// URIs directly
    df = pd.read_csv(gcs_uri, storage_options={"project": project_id})

    logger.info("File loaded", rows=len(df), columns=list(df.columns))

    df.to_parquet(raw_data.path + ".parquet", index=False)

    logger.info("Artifact written", path=raw_data.path)
