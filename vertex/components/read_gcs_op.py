"""
READ GCS COMPONENT
==================
Step 1 — Reads a CSV file from GCS and saves it as a Dataset artifact.

No transformation is applied here. The raw DataFrame is forwarded to
the preprocess component.
"""

from kfp.dsl import component, Output, Dataset

_ML_TRAINING_IMAGE = "europe-west3-docker.pkg.dev/prj-ceva-gr-wkfc-adapt-dev/gr-forecast/ml-training:latest"


@component(base_image=_ML_TRAINING_IMAGE)
def read_gcs_op(
    project_id: str,
    gcs_uri: str,
    raw_data: Output[Dataset],
):
    """Read a CSV from GCS and persist it as a Dataset artifact."""
    import logging
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Reading CSV from GCS: %s", gcs_uri)
    df = pd.read_csv(gcs_uri, storage_options={"project": project_id})
    logger.info("Read %d rows. Columns: %s", len(df), list(df.columns))

    df.to_parquet(raw_data.path + ".parquet", index=False)
    logger.info("Saved raw data to artifact: %s", raw_data.path)
