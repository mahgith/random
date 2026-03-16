"""
FORECASTING PIPELINE
====================
This file wires all components together into a single KFP pipeline.

HOW TO READ THIS FILE
---------------------
1. The @pipeline decorator registers the function as a KFP pipeline.
2. Inside the function, each component is called like a normal Python function.
3. Component outputs are passed as inputs to the next component — this is how
   KFP knows the execution order and data flow.
4. No actual computation happens here; this is just the graph definition.

PIPELINE GRAPH
--------------
data_ingestion
    └─► preprocessing
            ├─► training ─► evaluation ─► model_registration
            └─► evaluation (test_dataset)
"""

import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import pipeline

# Import all components
from pipelines.components.data_ingestion import data_ingestion_op
from pipelines.components.preprocessing import preprocessing_op
from pipelines.components.training import training_op
from pipelines.components.evaluation import evaluation_op
from pipelines.components.model_registration import model_registration_op


@pipeline(
    name="forecasting-pipeline",
    description="End-to-end forecasting MLOps pipeline on Vertex AI",
)
def forecasting_pipeline(
    # These are the pipeline-level parameters you set when you trigger a run.
    # They flow down into individual components.
    project_id: str,
    region: str,
    artifact_bucket: str,
    model_display_name: str,
    serving_container_image_uri: str,
    experiment_name: str,
    lookback_days: int = 365,
    forecast_horizon: int = 30,
    test_size_fraction: float = 0.2,
    mae_threshold: float = 10.0,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
):
    # ── Step 1: Data Ingestion ────────────────────────────────────────────────
    # Call the component function. KFP returns a "task" object.
    # Access outputs via: ingest_task.outputs["raw_dataset"]
    ingest_task = data_ingestion_op(
        project_id=project_id,
        raw_data_gcs_path=f"{artifact_bucket}/data/raw/",
        lookback_days=lookback_days,
    )
    # Set the machine type for this step (optional — defaults to n1-standard-4)
    ingest_task.set_cpu_limit("2")
    ingest_task.set_memory_limit("8G")

    # ── Step 2: Preprocessing ─────────────────────────────────────────────────
    preprocess_task = preprocessing_op(
        raw_dataset=ingest_task.outputs["raw_dataset"],
        forecast_horizon=forecast_horizon,
        test_size_fraction=test_size_fraction,
    )

    # ── Step 3: Training ──────────────────────────────────────────────────────
    train_task = training_op(
        train_dataset=preprocess_task.outputs["train_dataset"],
        project_id=project_id,
        experiment_name=experiment_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    # Request more resources for training if needed
    train_task.set_cpu_limit("4")
    train_task.set_memory_limit("16G")

    # ── Step 4: Evaluation ────────────────────────────────────────────────────
    eval_task = evaluation_op(
        model=train_task.outputs["model"],
        test_dataset=preprocess_task.outputs["test_dataset"],
        mae_threshold=mae_threshold,
    )

    # ── Step 5: Model Registration ────────────────────────────────────────────
    # Only runs if evaluation approved the model.
    # We pass the approval_decision file path from eval_task.
    register_task = model_registration_op(
        model=train_task.outputs["model"],
        approval_decision_path=eval_task.outputs["approval_decision"],
        project_id=project_id,
        region=region,
        model_display_name=model_display_name,
        serving_container_image_uri=serving_container_image_uri,
        pipeline_run_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
    )
    # Registration only makes sense after evaluation passes
    register_task.after(eval_task)


if __name__ == "__main__":
    # Compile the pipeline to a JSON file that Vertex AI can execute.
    # Run: python -m pipelines.pipeline.forecasting_pipeline
    from kfp.v2.compiler import Compiler
    import os

    output_path = "pipelines/compiled/forecasting_pipeline.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    Compiler().compile(
        pipeline_func=forecasting_pipeline,
        package_path=output_path,
    )
    print(f"Pipeline compiled to: {output_path}")
