from typing import Dict, Any
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics

# Docker image configuration
PROPHET = "europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images/prophet:latest"

@dsl.component(base_image=PROPHET)
def prophet_model(
    input_data: Input[Dataset],
    model_output: Output[Model],
    metrics: Output[Metrics],
    infra: Dict[str, Any],
    job_params: Dict[str, Any],
) -> None:
    """
    Trains and tunes a Prophet forecasting candidate model using Time Series Cross Validation.

    This component receives the TRAINING dataset (with the validation gate days already 
    removed). It performs hyperparameter tuning via Grid Search and runs Prophet's native 
    Cross Validation to assess robustness. The best parameters are selected based on the 
    lowest WAPE (Weighted Absolute Percentage Error), which is ideal for volume/logistics. 
    Finally, it trains a candidate model using these best parameters on the provided 
    training data, passing it forward for evaluation against the Champion.

    Args:
        input_data (Input[Dataset]): 
            The training dataset in CSV format (must contain 'ds' and 'y' columns).
        model_output (Output[Model]): 
            The physical artifact path where the candidate JSON model will be stored.
        metrics (Output[Metrics]): 
            The artifact where performance metrics of the candidate model will be logged.
        infra (Dict[str, Any]): 
            Global infrastructure configuration (project_id, location).
        job_params (Dict[str, Any]): 
            Pipeline and algorithm parameters including the hyperparameter grid.

    Returns:
        None.
    """
    
    import os
    import itertools
    import structlog
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from prophet import Prophet
    from prophet.serialize import model_to_json    
    from prophet.diagnostics import cross_validation
    from sklearn.metrics import (
        mean_absolute_percentage_error,
        mean_absolute_error,
        mean_squared_error
    )
    from google.cloud import aiplatform
    from common.core.logger import get_logger
    from common.prophet.helper import compute_forecast_metrics

    # Initialize the application structured logger
    logger = get_logger("training-tuning-task")

    try:
        # Unpack infrastructure and job configuration parameters
        project_id = infra["project_id"]
        location = infra["location"]
        grid_config = job_params["hyperparameter_grid"]
        experiment_name = job_params["experiment_name"]
        run_prefix = job_params["run_prefix"]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Bind context variables for GCP Cloud Logging
        structlog.contextvars.bind_contextvars(project_id=project_id)

        # Initialize Vertex AI SDK for experiment tracking
        aiplatform.init(project=project_id, location=location, 
                        experiment=f"{experiment_name}-{timestamp}")

        # Load the TRAINING dataset (Output from the Splitter)
        df_train = pd.read_csv(input_data.path)
        df_train = df_train.sort_values(by='ds').reset_index(drop=True)

        seasonal_period = 7 if grid_config.get("weekly_seasonality") == True else 1

        # Exhaustive Search (Grid Search)
        best_wape = float("inf")
        best_params = {}
        best_metrics = {}

        keys = grid_config.keys()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*grid_config.values())]
        
        logger.info("Evaluating combinations of hyperparameters.", numbers_of_combination=len(combinations))

        for i, params in enumerate(combinations):
            # Run unique identifier
            run_name = f"{run_prefix}-{timestamp}-{i}"
            
            # Start tracking in Vertex AI Experiments
            aiplatform.start_run(run=run_name)
            aiplatform.log_params({**params, "dataset_uri": input_data.uri})

            # Cross Validation inside the Training Set
            m = Prophet(**params)
            m.fit(df_train) 

            # Execute cross validation
            df_cv = cross_validation(m, initial='365 days', period='90 days', 
                                        horizon='30 days', parallel="threads")
            
            m_dict = compute_forecast_metrics(
                y_true=df_cv["y"], 
                y_pred=df_cv["yhat"], 
                seasonality_period=seasonal_period
            )

            # Extract WAPE for model selection
            wape = m_dict["wape"]
            
            # Log all needed metrics in Vertex AI Experiments
            aiplatform.log_metrics({
                "val_mape": m_dict["mape"],
                "val_mae": m_dict["mae"],
                "val_rmse": m_dict["rmse"],
                "val_wape": wape,
                "val_smape": m_dict["smape"],
                "val_mase": m_dict["mase"]
            }) 

            # Evaluate against the best WAPE seen so far
            if wape < best_wape:
                best_wape = wape
                best_params = params
                best_metrics = m_dict

            # Close the current Vertex AI run
            aiplatform.end_run()

        logger.info("Tuning completed", best_params=best_params, best_val_wape=best_wape)

        # Train the candidate model with the winning parameters on the TRAINING data
        logger.info("Training candidate model on training data with best parameters")
        candidate_model = Prophet(**best_params)
        candidate_model.fit(df_train)

        # Save the physical artifact for the Evaluator component
        os.makedirs(model_output.path, exist_ok=True)
        model_file = os.path.join(model_output.path, "model.json")
        with open(model_file, "w") as fout:
            fout.write(model_to_json(candidate_model))

        # Saving best metrics to the pipeline output
        metrics.log_metric("val_wape", best_metrics.get("wape", best_wape))
        metrics.log_metric("val_mape", best_metrics.get("mape"))
        metrics.log_metric("val_mae", best_metrics.get("mae"))
        metrics.log_metric("val_rmse", best_metrics.get("rmse"))
        metrics.log_metric("val_smape", best_metrics.get("smape"))
        metrics.log_metric("val_mase", best_metrics.get("mase"))

        # Saving best hyperparameters in metadata to pass to the Refit component later
        for param_name, param_value in best_params.items():
            model_output.metadata[param_name] = str(param_value)

        model_output.metadata["primary_metric"] = "val_wape"
        model_output.metadata["framework"] = "prophet"
        logger.info("Candidate model saved.", artifact_dir=model_output.path)

    except Exception as e:
        logger.error("Failure during training", error=str(e))
        # Ensure that if it fails, the active run is closed to avoid corrupting the experiment
        try:
            aiplatform.end_run()
        except Exception:
            pass
        raise RuntimeError(f"Failure during training: {e}") from e