from typing import Dict, Any
from kfp import dsl
from components.data.ingest.volume_forecast import extract_data_from_bq
from components.data.split.volume_forecast import split_time_series_data
from components.ml.train.volume_forecast import prophet_model
from components.ml.evaluate.volume_forecast import evaluate_prophet_model
from components.ml.compare.volume_forecast import gate_champion_vs_challenger
from components.ml.refit.volume_forecast import refit_prophet_model
from components.governance.register.volume_forecast import register_prophet_model

@dsl.pipeline(
    name="Volume forecast",
    description="Pipeline for forecasting volume per day"
)
def volume_forecast_pipeline(
        infra: Dict[str, Any],
        params_extract_data_from_bq: Dict[str, Any],
        params_split_time_series: Dict[str, Any],  
        params_train_prophet_model: Dict[str, Any],  
        params_evaluation_model: Dict[str, Any],   
        params_gate_champion_vs_challenger: Dict[str, Any],  
        params_register_prophet_model: Dict[str, Any],       
):
    # Task for ingestion of volume (LIGHT)
    # The task is executed in BigQuery with the retrieval of an 18-month
    # dataset of time series data. BigQuery engine handles the heavy lifting.
    # Machine provisioned by Vertex: 'e2-standard-2' (2 vCPUs, 8 GB RAM)
    # to meet the KFP Executor minimum baseline requirements.
    ingest_volume_task = extract_data_from_bq(
        infra=infra,
        job_params=params_extract_data_from_bq
    ) # type: ignore  
    ingest_volume_task.set_display_name("Ingest of volumes")
    ingest_volume_task.set_caching_options(enable_caching=False)
    ingest_volume_task.set_cpu_limit("2")
    ingest_volume_task.set_memory_limit("8G")

    # Task for splitting data (LIGHT)
    # Loads the small CSV dataset into a Pandas DataFrame in memory 
    # and splits it into training and validation (holdout) sets.
    # Machine provisioned by Vertex: 'e2-standard-2' (2 vCPUs, 8 GB RAM)
    # to meet the KFP Executor minimum baseline requirements.
    split_task = split_time_series_data(
        input_data=ingest_volume_task.outputs["dataset_output"],
        infra=infra,
        job_params=params_split_time_series
    ) # type: ignore  
    split_task.set_display_name("Splitting data")
    split_task.set_caching_options(enable_caching=False)
    split_task.set_cpu_limit("2")
    split_task.set_memory_limit("8G") 

    # Task for training model (HEAVY)
    # Performs hyperparameter tuning using Grid Search and Cross-Validation.
    # Prophet is highly CPU-bound, so multiple cores reduce hours to minutes.
    # Machine provisioned by Vertex: 'e2-standard-8' (8 vCPUs, 32 GB RAM).
    # (Vertex automatically rounds up RAM to match the standard 8-core template)
    train_task = prophet_model(
        input_data=split_task.outputs["training_data"],
        infra=infra,
        job_params=params_train_prophet_model
    ) # type: ignore
    train_task.set_display_name("Training prophet model")
    train_task.after(split_task)
    train_task.set_caching_options(enable_caching=False)
    train_task.set_cpu_limit("8")
    train_task.set_memory_limit("16G")    

    # Task for evaluation model (LIGHT)
    # Generates predictions over a short 30-day validation window and computes 
    # business metrics (WAPE, RMSE, Bias) to check against static thresholds.
    # Machine provisioned by Vertex: 'e2-standard-2' (2 vCPUs, 8 GB RAM)
    # to meet the KFP Executor minimum baseline requirements.
    evaluation_model_task = evaluate_prophet_model(
        model_input=train_task.outputs["model_output"],
        eval_data=split_task.outputs["validation_gate_data"],
        infra=infra,
        job_params=params_evaluation_model
    ) # type: ignore
    evaluation_model_task.set_display_name("Gate thresholds holdout")
    evaluation_model_task.after(train_task)
    evaluation_model_task.set_caching_options(enable_caching=False)   
    evaluation_model_task.set_cpu_limit("2")
    evaluation_model_task.set_memory_limit("8G")

    with dsl.If(evaluation_model_task.outputs["Output"] == True, name="gate-thresholds-passed"):

        # Task for champion vs challenger comparison (LIGHT)
        # Downloads the production Champion model JSON from GCS and compares 
        # its predictions mathematically against the Challenger.
        # Machine provisioned by Vertex: 'e2-standard-2' (2 vCPUs, 8 GB RAM)
        # to meet the KFP Executor minimum baseline requirements.
        champion_challenger_task = gate_champion_vs_challenger(
            candidate_model=train_task.outputs["model_output"],
            eval_data=split_task.outputs["validation_gate_data"],
            infra=infra,
            job_params=params_gate_champion_vs_challenger
        ) # type: ignore
        champion_challenger_task.set_display_name("Gate champion vs challenger")
        champion_challenger_task.set_caching_options(enable_caching=False) 
        champion_challenger_task.set_cpu_limit("2") 
        champion_challenger_task.set_memory_limit("8G")         

        with dsl.If(champion_challenger_task.outputs["Output"] == True, name="gate-champion-passed"):

            # Task for final model refit (MEDIUM)
            # Trains a single, final Prophet model using the winning parameters 
            # on 100% of the historical dataset before production deployment.
            # Machine provisioned by Vertex: 'e2-standard-2' (2 vCPUs, 8 GB RAM)
            refit_prophet_model_task = refit_prophet_model(
                full_data=ingest_volume_task.outputs["dataset_output"],
                candidate_model=train_task.outputs["model_output"],
                infra=infra,
            ) # type: ignore
            refit_prophet_model_task.set_display_name("Refit model (100% data)")
            refit_prophet_model_task.set_caching_options(enable_caching=False)
            refit_prophet_model_task.set_cpu_limit("2")
            refit_prophet_model_task.set_memory_limit("8G")     

            # Task for registering model (ULTRA-LIGHT)
            # Performs lightweight GAPIC and SDK API calls to Vertex AI Model Registry 
            # to upload the model artifact and handle lifecycle metadata/labels.
            # Machine provisioned by Vertex: 'e2-standard-2' (2 vCPUs, 8 GB RAM)
            # to meet the KFP Executor minimum baseline requirements.
            register_model_task = register_prophet_model(
                model_input=refit_prophet_model_task.outputs["refit_model"],
                infra=infra,
                job_params=params_register_prophet_model
            ) # type: ignore
            register_model_task.set_display_name("Register model")
            register_model_task.after(refit_prophet_model_task)
            register_model_task.set_caching_options(enable_caching=False)
            register_model_task.set_cpu_limit("2")
            register_model_task.set_memory_limit("8G")      