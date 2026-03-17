from typing import Dict, Any
from kfp import dsl
from kfp.dsl import Dataset, Input, Output

# Docker image configuration
BASE = "europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images/base:latest"

@dsl.component(base_image=BASE)
def split_time_series_data(
    input_data: Input[Dataset],
    training_data: Output[Dataset],
    validation_gate_data: Output[Dataset],
    infra: Dict[str, Any],
    job_params: Dict[str, Any]
):
    """
    Splits a chronological time series dataset into training and validation sets.

    This component reads an input dataset, sorts it chronologically by the 'ds' 
    date column, and splits it into two distinct datasets based on a specified 
    number of days from the end of the series. The older data is routed to the 
    training artifact, while the most recent days are isolated in the validation 
    artifact for blind testing (validation gate). It also binds context variables 
    to a structured logger for enhanced traceability in Cloud Logging.

    Args:
        input_data (Input[Dataset]): 
            The input time series dataset in CSV format. Must contain a 'ds' column.
        train_data (Output[Dataset]): 
            The output artifact where the training portion of the data will be saved.
        validation_gate_data (Output[Dataset]): 
            The output artifact where the validation portion of the data (the last N days) 
            to use during the validation gates, will be saved.
        infra (Dict[str, Any]): 
            Global infrastructure configuration.
            - project_id (str): The GCP Project ID.
            - location (str): The GCP region for the job (e.g., 'europe-west9').
        job_params (Dict[str, Any]): 
            Pipeline execution parameters.
            - validation_gate_days_data (int): The number of days at the end of the 
                time series to reserve for the validation gate.

    Returns:
        None.
    """
    import structlog
    import pandas as pd
    from common.core.logger import get_logger

    # Initialize the application structured logger
    logger = get_logger("splitting-task")

    try:

        # Unpack infrastructure and job configuration parameters into local variables
        project_id = infra["project_id"]
        location = infra["location"]
        validation_gate_days_data = job_params["validation_gate_days_data"]

        # Bind context variables. All subsequent logs will include these fields.
        # This is for filtering logs in GCP Cloud Logging.
        structlog.contextvars.bind_contextvars(
            project_id=project_id,
        )  

        logger.info(f"Getting {validation_gate_days_data} last day for blind validation", 
                    number_of_days=validation_gate_days_data)

        # Load and sort the data chronologically
        df = pd.read_csv(input_data.path)

        if "ds" not in df.columns or "y" not in df.columns:
            logger.error("Expected columns 'ds' and 'y' not found.", found_columns=list(df.columns))
            raise ValueError(f"Expected columns 'ds' and 'y' not found. Found columns: {list(df.columns)}")

        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values(by="ds").reset_index(drop=True)

        # Find the cut-off date
        max_date = df["ds"].max()
        cutoff_date = max_date - pd.Timedelta(days=validation_gate_days_data)

        # Cut the dataframe
        df_train = df[df["ds"] <= cutoff_date]
        df_test = df[df["ds"] > cutoff_date]

        logger.info(f"Training data: {len(df_train)} rows (up to {cutoff_date.date()})")
        logger.info(f"Test data: {len(df_test)} rows (from {cutoff_date.date()})")

        # Save datasets
        df_train.to_csv(training_data.path, index=False)
        df_test.to_csv(validation_gate_data.path, index=False)
        logger.info("Data prepared and saved in the output device.")

    except Exception as e:
        logger.error("Error separating the data", error=str(e))
        raise RuntimeError(f"Error separating the data: {e}") from e