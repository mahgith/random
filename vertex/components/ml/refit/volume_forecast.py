from typing import Dict, Any
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

# Docker image configuration
PROPHET = "europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images/prophet:latest"

@dsl.component(base_image=PROPHET)
def refit_prophet_model(
    full_data: Input[Dataset],
    candidate_model: Input[Model],
    refit_model: Output[Model],
    infra: Dict[str, Any],
) -> None:
    """
    Performs a final Refit of the Prophet model using 100% of the historical data.

    This component is triggered only if the candidate model successfully passes 
    both the threshold evaluation gate and the Champion vs. Challenger gate. It 
    extracts the winning hyperparameters stored in the candidate model's metadata and 
    trains a brand new Prophet model using the complete dataset (including the most 
    recent validation days). This ensures the production model is trained on the 
    absolute latest logistical patterns before being registered.

    Args:
        full_data (Input[Dataset]): 
            The complete historical dataset (100% of data) extracted from BigQuery.
        candidate_model (Input[Model]): 
            The KFP model artifact from the tuning phase, containing the best 
            hyperparameters in its metadata dictionary.
        refit_model (Output[Model]): 
            The physical artifact path where the newly refitted JSON model will be stored.
        infra (Dict[str, Any]): 
            Global infrastructure configuration (project_id).

    Returns:
        None.
    """
    
    import os
    import ast
    import structlog
    import pandas as pd
    from prophet import Prophet
    from prophet.serialize import model_to_json    
    from common.core.logger import get_logger

    # Initialize the application structured logger
    logger = get_logger("refit-prophet-task")

    try:
        # Unpack infrastructure configuration
        project_id = infra["project_id"]
        
        # Bind context variables for GCP Cloud Logging
        structlog.contextvars.bind_contextvars(project_id=project_id)

        logger.info("Starting final model refit with 100% of historical data.")

        # Extract winning hyperparameters from metadata
        best_params = {}
        for param_name, param_value in candidate_model.metadata.items():
            # Skip KFP standard tracking metadata
            if param_name in ["primary_metric", "framework"]:
                continue
            
            # Safely cast string metadata back to their original Python types (float, int, bool)
            try:
                best_params[param_name] = ast.literal_eval(param_value)
            except (ValueError, SyntaxError):
                best_params[param_name] = param_value  # Fallback to string if it's already a plain string
                
        logger.info("Hyperparameters successfully recovered.", best_params=best_params)

        # Load the COMPLETE dataset and apply Fail-Fast logic
        df_full = pd.read_csv(full_data.path)

        if "ds" not in df_full.columns or "y" not in df_full.columns:
            logger.error("Expected columns 'ds' and 'y' not found.", found_columns=list(df_full.columns))
            raise ValueError(f"Expected columns 'ds' and 'y' not found. Found columns: {list(df_full.columns)}")

        df_full = df_full.sort_values(by="ds").reset_index(drop=True)
        
        logger.info(f"Full dataset loaded. Total rows: {len(df_full)}. Date range: {df_full['ds'].min()} to {df_full['ds'].max()}")

        # Train the ultimate production model
        logger.info("Fitting Prophet model on full history...")
        final_m = Prophet(**best_params)
        final_m.fit(df_full)

        # Save the final artifact for the Registry component
        os.makedirs(refit_model.path, exist_ok=True)
        model_file = os.path.join(refit_model.path, "model.json")
        with open(model_file, "w") as fout:
            fout.write(model_to_json(final_m))

        # Transfer metadata to the final model artifact
        for k, v in candidate_model.metadata.items():
            refit_model.metadata[k] = v
            
        refit_model.metadata["is_refitted"] = "True"

        logger.info("Refit completed successfully. Artifact ready for registration.", artifact_dir=refit_model.path)

    except Exception as e:
        logger.error("Failure during model refit", error=str(e))
        raise RuntimeError(f"Failure during model refit: {e}") from e