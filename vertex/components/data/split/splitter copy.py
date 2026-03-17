from kfp import dsl
from kfp.dsl import Dataset, Input, Output
from typing import Dict, Any

IMAGE = "europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images/base:latest"

@dsl.component(base_image=IMAGE)
def split_time_series_data(
    input_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    job_params: Dict[str, Any]
):
    import structlog
    import pandas as pd
    from common.core.logger import get_logger

    # Initialize the application structured logger
    logger = get_logger("split-training-task")

    try:

        # Unpack infrastructure and job configuration parameters into local variables
        test_days = job_params["test_days"]
        logger.info(f"Getting {test_days} last day for blind validation", number_of_days=test_days)

        # Load and sort the data chronologically
        df = pd.read_csv(input_data.path)
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values(by="ds").reset_index(drop=True)

        # Find the cut-off date
        max_date = df["ds"].max()
        cutoff_date = max_date - pd.Timedelta(days=test_days)

        # Cut the dataframe
        df_train = df[df["ds"] <= cutoff_date]
        df_test = df[df["ds"] > cutoff_date]

        logger.info(f"Training data: {len(df_train)} rows (up to {cutoff_date.date()})")
        logger.info(f"Test data: {len(df_test)} rows (from {cutoff_date.date()})")

        # Save datasets
        df_train.to_csv(train_data.path, index=False)
        df_test.to_csv(test_data.path, index=False)
        logger.info("Data prepared and saved in the output device.")

    except Exception as e:
        logger.error("Error separating the data", error=str(e))
        raise RuntimeError(f"Error separating the data: {e}") from e