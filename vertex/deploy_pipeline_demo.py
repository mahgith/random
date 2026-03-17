from kfp import compiler
from google.cloud import aiplatform
from vertex.common.core.logger import get_logger
from demo.vertex.pipelines.volume_forecast import volume_forecast_pipeline
from demo.vertex.common.core.load_kfp_params import load_kfp_params

def deploy():

    # Initialize the application structured logger
    logger = get_logger("deploy-script")

    pipeline_json = "demo/vertex/pipelines/templates/volume_forecast.json"

    # Generate the file for Vertex AI.
    compiler.Compiler().compile(
        pipeline_func=volume_forecast_pipeline,
        package_path=pipeline_json
    )
    logger.info("Pipeline compiled successfully", file=pipeline_json)


    # Loading parameters for the pipeline
    full_params = load_kfp_params("volume_forecast", "params_v1")
    infra = full_params.get("infra", {})
    params_extract_data_from_bq = full_params.get("params_extract_data_from_bq", {})
    params_split_time_series = full_params.get("params_split_time_series", {})
    params_train_prophet_model = full_params.get("params_train_prophet_model", {})
    params_evaluation_model = full_params.get("params_evaluation_model", {})
    params_gate_champion_vs_challenger = full_params.get("params_gate_champion_vs_challenger", {})
    params_register_prophet_model = full_params.get("params_register_prophet_model", {})    

    # Parameter definition
    parameter_values = {
        "infra": infra,
        "params_extract_data_from_bq": params_extract_data_from_bq,
        "params_split_time_series": params_split_time_series,
        "params_train_prophet_model": params_train_prophet_model,
        "params_evaluation_model": params_evaluation_model,
        "params_gate_champion_vs_challenger": params_gate_champion_vs_challenger,
        "params_register_prophet_model": params_register_prophet_model
    }

    # Vertex AI Initialization
    aiplatform.init(
        project=infra["project_id"],
        location=infra["location"],
        staging_bucket=infra["bucket_name"]
    )
    

    logger.info("Submitting pipeline to Vertex AI", 
                project=infra["project_id"],
                location=infra["location"],
                bucket=infra["bucket_name"])

    # Job ejecution
    job = aiplatform.PipelineJob(
        display_name="volume-forecast",
        template_path=pipeline_json,
        pipeline_root=f"gs://{infra["bucket_name"]}/pipeline_artifacts",
        parameter_values=parameter_values,
        enable_caching=False,
    )

    job.submit()
    
    # Print the direct link to view the progress in the GCP console
    logger.info("Pipeline submitted successfully")
    print(f"🔗 Dashboard: {job._dashboard_uri()}")

if __name__ == "__main__":
    deploy()