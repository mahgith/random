"""
FORECASTING PIPELINE
====================
Wires all components into a single KFP pipeline parameterised by `direction`.

PIPELINE GRAPH
--------------
data_ingestion
    └─► preprocessing
            └─► training ─► evaluation ─► model_registration

USAGE
-----
Compile and run for a specific direction:

    python scripts/run_pipeline.py --direction inbound
    python scripts/run_pipeline.py --direction outbound

Each direction is an independent pipeline run.  Monitoring / retraining
one direction never touches the other.

SCHEDULING
----------
Set up two Vertex AI Pipeline Schedules (one per direction) with the
parameter dict from configs.settings.build_pipeline_params().
If outbound degrades, cancel/retrigger only the outbound schedule.
"""

from kfp.v2 import dsl
from kfp.v2.dsl import pipeline

from pipelines.components.data_ingestion import data_ingestion_op
from pipelines.components.preprocessing import preprocessing_op
from pipelines.components.training import training_op
from pipelines.components.evaluation import evaluation_op
from pipelines.components.model_registration import model_registration_op


@pipeline(
    name="package-volume-forecasting",
    description="Inbound/outbound package volume forecasting — 7-28 day horizon",
)
def forecasting_pipeline(
    # ── Identity ─────────────────────────────────────────────────────────────
    direction: str,                        # "inbound" or "outbound"

    # ── GCP ──────────────────────────────────────────────────────────────────
    project_id: str,
    region: str,
    artifact_bucket: str,
    experiment_name: str,

    # ── Data source ───────────────────────────────────────────────────────────
    bq_tables_json: str,                   # JSON list of {dataset, table, ...}
    bq_columns_json: str,                  # JSON dict of logical → BQ column names

    # ── L1 baseline ───────────────────────────────────────────────────────────
    lookback_days: int,
    half_life_days: int,

    # ── L2B Prophet ───────────────────────────────────────────────────────────
    prophet_changepoint_prior_scale: float,

    # ── L3 LightGBM ───────────────────────────────────────────────────────────
    lgbm_n_estimators: int,
    lgbm_learning_rate: float,
    lgbm_num_leaves: int,

    # ── Evaluation ────────────────────────────────────────────────────────────
    forecast_horizon: int,
    evaluation_start_date: str,
    backtest_step_days: int,
    mape_threshold: float,

    # ── Registration ──────────────────────────────────────────────────────────
    model_display_name: str,
    serving_container_image_uri: str = "",  # leave empty until serving image ready
):
    # ── Step 1: Data Ingestion ────────────────────────────────────────────────
    ingest_task = data_ingestion_op(
        direction=direction,
        project_id=project_id,
        bq_tables_json=bq_tables_json,
        bq_columns_json=bq_columns_json,
    )
    ingest_task.set_cpu_limit("2").set_memory_limit("8G")

    # ── Step 2: Preprocessing ─────────────────────────────────────────────────
    preprocess_task = preprocessing_op(
        direction=direction,
        raw_dataset=ingest_task.outputs["raw_dataset"],
    )
    preprocess_task.set_cpu_limit("2").set_memory_limit("8G")

    # ── Step 3: Training ──────────────────────────────────────────────────────
    train_task = training_op(
        direction=direction,
        processed_data=preprocess_task.outputs["processed_data"],
        lookback_days=lookback_days,
        half_life_days=half_life_days,
        prophet_changepoint_prior_scale=prophet_changepoint_prior_scale,
        lgbm_n_estimators=lgbm_n_estimators,
        lgbm_learning_rate=lgbm_learning_rate,
        lgbm_num_leaves=lgbm_num_leaves,
    )
    train_task.set_cpu_limit("8").set_memory_limit("32G")

    # ── Step 4: Evaluation ────────────────────────────────────────────────────
    eval_task = evaluation_op(
        direction=direction,
        processed_data=preprocess_task.outputs["processed_data"],
        model=train_task.outputs["model"],
        evaluation_start_date=evaluation_start_date,
        forecast_horizon=forecast_horizon,
        backtest_step_days=backtest_step_days,
        mape_threshold=mape_threshold,
    )
    eval_task.set_cpu_limit("4").set_memory_limit("16G")

    # ── Step 5: Model Registration ────────────────────────────────────────────
    register_task = model_registration_op(
        direction=direction,
        model=train_task.outputs["model"],
        approval_decision_path=eval_task.outputs["approval_decision"],
        project_id=project_id,
        region=region,
        model_display_name=model_display_name,
        experiment_name=experiment_name,
        pipeline_run_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        serving_container_image_uri=serving_container_image_uri,
    )
    register_task.after(eval_task)


if __name__ == "__main__":
    from kfp.v2.compiler import Compiler
    import os

    output_path = "pipelines/compiled/forecasting_pipeline.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Compiler().compile(pipeline_func=forecasting_pipeline, package_path=output_path)
    print(f"Pipeline compiled to: {output_path}")
