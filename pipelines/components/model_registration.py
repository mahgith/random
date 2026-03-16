"""
MODEL REGISTRATION COMPONENT
=============================
Responsibility: Register the approved model in Vertex AI Model Registry.

What happens here:
  - Reads the model artifact produced by training_op
  - Uploads it to Vertex AI Model Registry with metadata
  - Optionally triggers deployment to a Vertex AI Endpoint

Vertex AI Model Registry is the central store for your models.
From there you can deploy to online endpoints (real-time prediction)
or batch prediction jobs.

IMPORTANT — Serving Container
------------------------------
The serving_container_image_uri must match your model type:
  sklearn:  us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest
  xgboost:  us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest
  lightgbm: us-docker.pkg.dev/vertex-ai/prediction/lightgbm-cpu.3-3:latest
  custom:   Your own container (needed for complex pre/post processing)
"""

from kfp.v2.dsl import component, Input, Output, Model


@component(
    packages_to_install=[
        "google-cloud-aiplatform",
    ],
    base_image="python:3.10-slim",
)
def model_registration_op(
    # ── Inputs ────────────────────────────────────────────────────────────────
    model: Input[Model],
    approval_decision_path: str,          # path to the approval_decision file
    project_id: str,
    region: str,
    model_display_name: str,
    serving_container_image_uri: str,
    pipeline_run_name: str = "",          # injected by the pipeline for traceability

    # ── Outputs ───────────────────────────────────────────────────────────────
    registered_model_resource_name: Output[str] = None,   # type: ignore[assignment]
):
    """Register the model in Vertex AI Model Registry if approved."""
    import logging
    from google.cloud import aiplatform

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # ── Check approval ────────────────────────────────────────────────────────
    with open(approval_decision_path) as f:
        decision = f.read().strip()

    if decision != "approved":
        log.warning(f"Model was NOT approved (decision='{decision}'). Skipping registration.")
        # Write empty output so downstream steps don't fail
        with open(registered_model_resource_name.path, "w") as f:
            f.write("")
        return

    # ── Register model ────────────────────────────────────────────────────────
    log.info(f"Registering model '{model_display_name}' in Vertex AI ...")
    aiplatform.init(project=project_id, location=region)

    uploaded_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model.uri,           # GCS path of the model directory
        serving_container_image_uri=serving_container_image_uri,
        labels={
            "pipeline_run": pipeline_run_name.replace("/", "_")[:63],
            "framework": "sklearn",       # update to match your framework
        },
    )

    resource_name = uploaded_model.resource_name
    log.info(f"Model registered: {resource_name}")

    # Write the resource name so downstream steps (e.g. deployment) can use it
    with open(registered_model_resource_name.path, "w") as f:
        f.write(resource_name)

    # ── OPTIONAL: Deploy to endpoint ─────────────────────────────────────────
    # Uncomment when you're ready to serve predictions.
    #
    # endpoint = aiplatform.Endpoint.create(
    #     display_name=f"{model_display_name}-endpoint",
    #     project=project_id,
    #     location=region,
    # )
    # uploaded_model.deploy(
    #     endpoint=endpoint,
    #     machine_type="n1-standard-2",
    #     min_replica_count=1,
    #     max_replica_count=3,
    # )
    # log.info(f"Model deployed to endpoint: {endpoint.resource_name}")
