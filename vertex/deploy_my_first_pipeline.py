from kfp import compiler
from google.cloud import aiplatform
from demo.vertex.common.core.logger import get_logger
from demo.vertex.pipelines.my_first_pipeline import my_first_pipeline
from demo.vertex.common.core.load_kfp_params import load_kfp_params

def deploy():

    # Initialize the application structured logger
    logger = get_logger("deploy-script")

    pipeline_json = "demo/vertex/pipelines/templates/my_first_pipeline.json"

    # Generate the file for Vertex AI.
    compiler.Compiler().compile(
        pipeline_func=my_first_pipeline,
        package_path=pipeline_json
    )
    logger.info("Pipeline compiled successfully", file=pipeline_json)


    # Loading parameters for the pipeline
    full_params = load_kfp_params("my_first_pipeline", "params_v1")
    infra = full_params.get("infra", {})
    params_ingestion_shipments = full_params.get("params_ingestion_shipments", {})
    params_ingestion_events = full_params.get("params_ingestion_events", {})
    params_clean_and_merge_tables = full_params.get("params_clean_and_merge_tables", {})
    params_build_training_dataset = full_params.get("params_build_training_dataset", {})
    # params_train_xgboost_model = full_params.get("params_train_xgboost_model", {})

    # Parameter definition
    parameter_values = {
        "infra": infra,
        "params_ingestion_shipments": params_ingestion_shipments,
        "params_ingestion_events": params_ingestion_events,
        "params_clean_and_merge_tables": params_clean_and_merge_tables,
        "params_build_training_dataset": params_build_training_dataset,
        # "params_train_xgboost_model": params_train_xgboost_model,
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
        display_name="my-first-pipeline",
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