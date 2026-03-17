from typing import Dict, Any
from kfp import dsl
from kfp.dsl import Input, Model

# Docker image configuration
BASE = "europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images/base:latest"

@dsl.component(base_image=BASE)
def register_prophet_model(
    model_input: Input[Model],
    infra: Dict[str, Any],
    job_params: Dict[str, Any],
):
    """
    Registers a validated Prophet model into the Vertex AI Model Registry and manages lifecycle labels.

    This component takes the physical model artifact (usually after a Refit process) and 
    uploads it to Vertex AI. If a model with the specified display name already exists, 
    it appends the new upload as the latest default version in the same model tree. 
    It assigns the target environment and 'champion' role labels dynamically based on 
    pipeline parameters. Finally, it uses the GAPIC client to iterate through all 
    historical versions of the model, actively demoting any previous champions by 
    updating their role to 'archived', ensuring a single source of truth in the registry.

    Args:
        model_input (Input[Model]): 
            The physical artifact containing the finalized Prophet model (JSON).
        infra (Dict[str, Any]): 
            Global infrastructure configuration (project_id, location).
        job_params (Dict[str, Any]): 
            Registration and metadata parameters:
            - model_display_name (str): The name of the model in the registry.
            - serving_image_uri (str): Container image required for online predictions.
            - champion_label_env (str): Target environment label (e.g., 'dev', 'prod').
            - champion_label_role (str): The active role label (e.g., 'champion').
            - archived_label_role (str, optional): The demoted role label (defaults to 'archived').

    Returns:
        None.
    """
    import structlog
    from google.cloud import aiplatform
    from google.cloud.aiplatform import gapic
    from google.protobuf import field_mask_pb2
    from common.core.logger import get_logger

    # Initialize the application structured logger
    logger = get_logger("register-prophet-task")

    try:
        # Unpack infrastructure and job configuration parameters into local variables
        project_id = infra["project_id"]
        location = infra["location"]
        model_display_name = job_params["model_display_name"]
        serving_image_uri = job_params["serving_image_uri"]
        env_label = job_params.get("champion_label_env", "dev")
        champion_role = job_params.get("champion_label_role", "champion")
        archived_role = job_params.get("archived_label_role", "archived")

        # Bind context variables
        structlog.contextvars.bind_contextvars(project_id=project_id)            

        # Initialize the Vertex AI SDK
        aiplatform.init(project=project_id, location=location)

        # Checking if a model with that name exists
        logger.info("Checking if the model already exists in the registry.", display_name=model_display_name)
        existing_model = aiplatform.Model.list(
            filter=f'display_name="{model_display_name}" '
        )

        parent_model_id = None
        if not existing_model:           
            logger.info("No previous model was found. A new model tree will be created.")
        else:
            parent_model = existing_model[0]
            parent_model_id = parent_model.resource_name
            logger.info("Parent model found. It will be registered as a new version.", parent_model_id=parent_model_id)

        # Register the model by pointing to the GCS path where Kubeflow left the artifact
        logger.info("Registering model from uri", uri=model_input.uri)

        # Ensuring that the new version is the main one and setting dynamic labels
        upload_kwargs = {
            "artifact_uri": model_input.uri,
            "serving_container_image_uri": serving_image_uri,  
            "description": "Prophet model for package volume estimation",
            "labels": {
                "env": env_label, 
                "role": champion_role
            },
            "is_default_version": True,
        }

        if parent_model_id:
            upload_kwargs["parent_model"] = parent_model_id
        else:
            upload_kwargs["display_name"] = model_display_name

        model = aiplatform.Model.upload(**upload_kwargs)
        logger.info("Model successfully registered", id=model.resource_name)

        # Cleaning labels in ALL old versions of this specific model
        logger.info("Scanning all historical versions to remove old champion labels.")
        
        client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        model_client = gapic.ModelServiceClient(client_options=client_options)

        # Fetch ALL versions of this parent model
        versions = model_client.list_model_versions(name=model.resource_name)

        cleaned_count = 0
        for version in versions:
            # Ignore the brand new version we just uploaded
            if version.version_id == model.version_id:
                continue

            # If we find ANY old version holding the crown, we strip it
            if version.labels.get("role") == champion_role:
                logger.info("Removing old champion label.", version_id=version.version_id)

                exact_version_path = f"{model.resource_name}@{version.version_id}"
                
                # Update with the archived role, preserving the environment
                version_resource = gapic.Model(
                    name=exact_version_path,
                    labels={"env": env_label, "role": archived_role}
                )
                
                update_mask = field_mask_pb2.FieldMask(paths=["labels"])
                model_client.update_model(model=version_resource, update_mask=update_mask)
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Complete cleanup. {cleaned_count} old version(s) archived.")
        else:
            logger.info("No old champions found. The registry is clean.")

    except Exception as e:
        logger.error("Failure during model registration", error=str(e))
        raise RuntimeError(f"Failure during model registration: {e}") from e