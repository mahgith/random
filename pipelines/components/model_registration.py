"""
MODEL REGISTRATION COMPONENT
=============================
Registers the refitted model bundle in Vertex AI Model Registry and
manages champion/archived lifecycle labels across all historical versions.

The bundle directory (produced by refit_op) contains:
    config.json              — direction, hyperparams, training metadata
    multiplier_table.parquet — L2A lookup
    prophet_model.pkl        — serialised Prophet model
    lgbm_model.joblib        — LightGBM model
    lgbm_features.json       — L3 feature list

Lifecycle management
--------------------
1. Upload the new model version with label role=champion
2. Scan all previous versions of the same model tree
3. Demote any old champion versions to role=archived
This ensures exactly one champion exists at any time.
"""

from kfp.v2.dsl import component, Input, Output, Model

_FORECASTING_IMAGE = "europe-west1-docker.pkg.dev/your-gcp-project-id/ml-images/forecasting:latest"


@component(base_image=_FORECASTING_IMAGE)
def model_registration_op(
    direction: str,
    model: Input[Model],
    champion_gate_decision_path: str,   # path to file written by champion_vs_challenger_op
    project_id: str,
    region: str,
    model_display_name: str,
    experiment_name: str,
    champion_label_env: str,
    champion_label_role: str,
    archived_label_role: str,
    serving_container_image_uri: str = "",
    registered_model_resource_name: Output[str] = None,  # type: ignore[assignment]
):
    """Register the refitted model and manage champion/archived labels."""
    import structlog
    from google.cloud import aiplatform
    from google.cloud.aiplatform import gapic
    from google.protobuf import field_mask_pb2
    from common.core.logger import get_logger

    logger = get_logger("model-registration")
    structlog.contextvars.bind_contextvars(
        direction=direction,
        project_id=project_id,
        model_display_name=model_display_name,
    )

    # ── Check champion gate decision ──────────────────────────────────────────
    with open(champion_gate_decision_path) as f:
        decision = f.read().strip()

    if decision != "approved":
        logger.warning(
            "Champion gate not passed — skipping registration",
            decision=decision,
        )
        with open(registered_model_resource_name.path, "w") as f:
            f.write("")
        return

    # ── Register ──────────────────────────────────────────────────────────────
    logger.info("Initialising Vertex AI SDK")
    aiplatform.init(project=project_id, location=region)

    existing = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
    parent_model_id = None
    if existing:
        parent_model_id = existing[0].resource_name
        logger.info("Existing model tree found — adding new version", parent_model_id=parent_model_id)
    else:
        logger.info("No existing model — creating new model tree")

    upload_kwargs = dict(
        artifact_uri=model.uri,
        description=f"Package volume forecasting — {direction}",
        labels={
            "env": champion_label_env,
            "role": champion_label_role,
            "direction": direction,
        },
        is_default_version=True,
    )

    if serving_container_image_uri:
        upload_kwargs["serving_container_image_uri"] = serving_container_image_uri
    else:
        # Placeholder so the API call succeeds; replace when serving image is built
        upload_kwargs["serving_container_image_uri"] = (
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
        )

    if parent_model_id:
        upload_kwargs["parent_model"] = parent_model_id
    else:
        upload_kwargs["display_name"] = model_display_name

    registered = aiplatform.Model.upload(**upload_kwargs)
    resource_name = registered.resource_name
    logger.info("Model registered", resource_name=resource_name)

    # ── Demote old champion versions → archived ───────────────────────────────
    logger.info("Scanning previous versions for old champion labels")
    client_options = {"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    model_client = gapic.ModelServiceClient(client_options=client_options)
    versions = model_client.list_model_versions(name=registered.resource_name)

    cleaned_count = 0
    for version in versions:
        if version.version_id == registered.version_id:
            continue
        if version.labels.get("role") == champion_label_role:
            logger.info("Archiving old champion", version_id=version.version_id)
            exact_path = f"{registered.resource_name}@{version.version_id}"
            version_resource = gapic.Model(
                name=exact_path,
                labels={"env": champion_label_env, "role": archived_label_role},
            )
            mask = field_mask_pb2.FieldMask(paths=["labels"])
            model_client.update_model(model=version_resource, update_mask=mask)
            cleaned_count += 1

    if cleaned_count:
        logger.info("Old champion versions archived", count=cleaned_count)
    else:
        logger.info("No old champions found — registry is clean")

    with open(registered_model_resource_name.path, "w") as f:
        f.write(resource_name)
