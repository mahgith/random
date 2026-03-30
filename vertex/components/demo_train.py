"""
Step 3 — Simulated training and prediction.

The "model" computes the mean of the target column and uses that single
constant as the prediction for every row. This proves the pipeline runs
end-to-end without needing any real ML code.

Outputs
-------
predictions : Dataset
    A parquet file with one prediction per input row, all equal to mean(target).
"""

from kfp.dsl import component, Input, Output, Dataset

BASE_IMAGE = "europe-west3-docker.pkg.dev/prj-ceva-gr-wkfc-adapt-dev/gr-forecast/base:latest"


@component(base_image=BASE_IMAGE)
def demo_train_op(
    processed_data: Input[Dataset],
    target_column: str,         # name of the column to predict
    predictions: Output[Dataset],
):
    import pandas as pd
    from common.core.logger import get_logger

    logger = get_logger("demo-train")

    df = pd.read_parquet(processed_data.path + ".parquet")

    # "Train": compute a single constant — the historical mean
    constant = float(df[target_column].mean())
    logger.info("Constant-mean model", target_column=target_column, constant=round(constant, 4))

    # "Predict": assign that constant to every row
    df["prediction"] = constant

    logger.info("Predictions generated", rows=len(df), constant_value=round(constant, 4))

    df[["prediction"]].to_parquet(predictions.path + ".parquet", index=False)

    logger.info("Artifact written", path=predictions.path)
