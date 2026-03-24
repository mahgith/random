"""
Step 2 — Simulated preprocessing.

Receives the raw data artifact from the previous step, logs some basic
statistics to show the data was received correctly, then passes it forward
unchanged.

In a real pipeline this is where you would clean, transform, and enrich the
data. For now it is intentionally empty.
"""

from kfp.dsl import component, Input, Output, Dataset

BASE_IMAGE = "europe-west9-docker.pkg.dev/prj-ceva-gr-wkfc-adapt-dev/gr-forecast/base:latest"


@component(base_image=BASE_IMAGE)
def demo_preprocess_op(
    raw_data: Input[Dataset],
    processed_data: Output[Dataset],
):
    import pandas as pd
    from common.core.logger import get_logger

    logger = get_logger("demo-preprocess")

    df = pd.read_parquet(raw_data.path + ".parquet")

    logger.info(
        "Received data — no processing applied in demo",
        rows=len(df),
        columns=list(df.columns),
    )

    # Pass the data forward as-is
    df.to_parquet(processed_data.path + ".parquet", index=False)

    logger.info("Artifact written", path=processed_data.path)
