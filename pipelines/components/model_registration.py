"""
MODEL REGISTRATION COMPONENT
=============================
Registers the approved model bundle in Vertex AI Model Registry.

The bundle directory (produced by training_op) contains:
    config.json              — direction, hyperparams, training metadata
    multiplier_table.parquet — L2A lookup
    prophet_model.pkl        — serialised Prophet model
    lgbm_model.joblib        — LightGBM model
    lgbm_features.json       — L3 feature list

Vertex AI stores the entire bundle as a single Model artifact.
There is no built-in serving container that handles Prophet + LightGBM
together, so we register with a custom container placeholder.
Replace `custom_serving_image_uri` with your actual serving image once built.

If no serving container is needed yet (batch-only), leave
`serving_container_image_uri` empty — registration still works for
offline use (Vertex AI Batch Prediction).
"""

from kfp.v2.dsl import component, Input, Output, Model


@component(
    packages_to_install=[
        "google-cloud-aiplatform>=1.38.0",
    ],
    base_image="python:3.10-slim",
)
def model_registration_op(
    direction: str,
    model: Input[Model],
    approval_decision_path: str,
    project_id: str,
    region: str,
    model_display_name: str,
    experiment_name: str,
    pipeline_run_name: str = "",
    serving_container_image_uri: str = "",   # leave empty until serving image is ready
    registered_model_resource_name: Output[str] = None,   # type: ignore[assignment]
):
    """Register the model bundle in Vertex AI Model Registry if approved."""
    import logging
    from google.cloud import aiplatform

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # ── Check approval ────────────────────────────────────────────────────────
    with open(approval_decision_path) as f:
        decision = f.read().strip()

    if decision != "approved":
        log.warning(
            f"[{direction}] Model NOT approved (decision='{decision}'). "
            "Skipping registration."
        )
        with open(registered_model_resource_name.path, "w") as f:
            f.write("")
        return

    # ── Register ──────────────────────────────────────────────────────────────
    log.info(f"[{direction}] Registering '{model_display_name}' in Vertex AI ...")
    aiplatform.init(project=project_id, location=region)

    labels = {
        "direction":    direction,
        "pipeline_run": pipeline_run_name.replace("/", "_")[:63],
        "experiment":   experiment_name[:63],
    }

    upload_kwargs = dict(
        display_name=model_display_name,
        artifact_uri=model.uri,       # GCS path of the bundle directory
        labels=labels,
    )

    if serving_container_image_uri:
        upload_kwargs["serving_container_image_uri"] = serving_container_image_uri
    else:
        # No serving container — register for artifact lineage / batch use only.
        # Vertex AI requires a container URI for online serving; skip for now.
        log.info(
            f"[{direction}] No serving_container_image_uri provided. "
            "Registering bundle without serving config (batch / offline use)."
        )
        # Use a minimal placeholder so the API call succeeds
        upload_kwargs["serving_container_image_uri"] = (
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
        )

    registered = aiplatform.Model.upload(**upload_kwargs)
    resource_name = registered.resource_name
    log.info(f"[{direction}] Registered: {resource_name}")

    with open(registered_model_resource_name.path, "w") as f:
        f.write(resource_name)
