"""
FORECASTING PIPELINE
====================
Wires all components into a single KFP pipeline parameterised by `direction`.

PIPELINE GRAPH
--------------
data_ingestion
    └─► preprocessing
            └─► training  (HP grid search + Vertex AI Experiments)
                    └─► evaluation  (rolling backtest, WAPE + MAPE gates)
                            └─► champion_vs_challenger  (compare vs prod model)
                                    └─► refit  (train on 100% data with best params)
                                            └─► model_registration  (champion label management)

Gates
-----
  evaluation:             mean_wape ≤ wape_threshold AND mean_mape ≤ mape_threshold
  champion_vs_challenger: challenger_wape ≤ champion_wape - max(delta_abs, delta_rel × champion_wape)

If either gate is not passed, model_registration is skipped (writes an empty
resource name). The pipeline still completes successfully.

USAGE
-----
    python scripts/run_pipeline.py --direction inbound
    python scripts/run_pipeline.py --direction outbound
"""

from kfp.v2 import dsl
from kfp.v2.dsl import pipeline

from pipelines.components.data_ingestion import data_ingestion_op
from pipelines.components.preprocessing import preprocessing_op
from pipelines.components.training import training_op
from pipelines.components.evaluation import evaluation_op
from pipelines.components.champion_vs_challenger import champion_vs_challenger_op
from pipelines.components.refit import refit_op
from pipelines.components.model_registration import model_registration_op


@pipeline(
    name="package-volume-forecasting",
    description="Inbound/outbound package volume forecasting — 7-28 day horizon",
)
def forecasting_pipeline(
    # ── Identity ─────────────────────────────────────────────────────────────
    direction: str,

    # ── GCP ──────────────────────────────────────────────────────────────────
    project_id: str,
    region: str,
    artifact_bucket: str,

    # ── Data source ───────────────────────────────────────────────────────────
    bq_tables_json: str,
    bq_columns_json: str,

    # ── Training ──────────────────────────────────────────────────────────────
    lookback_days: int,
    half_life_days: int,
    hp_grid_json: str,
    tuning_num_cutoffs: int,
    experiment_name: str,
    run_prefix: str,

    # ── Evaluation ────────────────────────────────────────────────────────────
    forecast_horizon: int,
    evaluation_start_date: str,
    backtest_step_days: int,
    mape_threshold: float,
    wape_threshold: float,

    # ── Champion vs Challenger ────────────────────────────────────────────────
    model_display_name: str,
    champion_label_env: str,
    champion_label_role: str,
    champ_delta_rel: float,
    champ_delta_abs: float,

    # ── Registration ──────────────────────────────────────────────────────────
    serving_container_image_uri: str,
    archived_label_role: str,
):
    # ── Step 1: Ingest from BigQuery ──────────────────────────────────────────
    ingest_task = data_ingestion_op(
        direction=direction,
        project_id=project_id,
        bq_tables_json=bq_tables_json,
        bq_columns_json=bq_columns_json,
    )
    ingest_task.set_display_name("Ingest — BigQuery")
    ingest_task.set_caching_options(enable_caching=True)
    ingest_task.set_cpu_limit("2")
    ingest_task.set_memory_limit("8G")

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    preprocess_task = preprocessing_op(
        direction=direction,
        raw_dataset=ingest_task.outputs["raw_dataset"],
    )
    preprocess_task.set_display_name("Preprocessing — calendar + holiday features")
    preprocess_task.set_caching_options(enable_caching=True)
    preprocess_task.set_cpu_limit("2")
    preprocess_task.set_memory_limit("8G")

    # ── Step 3: HP tuning + candidate training ────────────────────────────────
    training_task = training_op(
        direction=direction,
        processed_data=preprocess_task.outputs["processed_data"],
        lookback_days=lookback_days,
        half_life_days=half_life_days,
        hp_grid_json=hp_grid_json,
        tuning_num_cutoffs=tuning_num_cutoffs,
        project_id=project_id,
        region=region,
        experiment_name=experiment_name,
        run_prefix=run_prefix,
    )
    training_task.set_display_name("Training — HP tuning + candidate model")
    training_task.set_caching_options(enable_caching=False)
    training_task.set_cpu_limit("8")
    training_task.set_memory_limit("32G")

    # ── Step 4: Rolling backtest — WAPE + MAPE gate ───────────────────────────
    evaluation_task = evaluation_op(
        direction=direction,
        processed_data=preprocess_task.outputs["processed_data"],
        model=training_task.outputs["model"],
        evaluation_start_date=evaluation_start_date,
        forecast_horizon=forecast_horizon,
        backtest_step_days=backtest_step_days,
        mape_threshold=mape_threshold,
        wape_threshold=wape_threshold,
    )
    evaluation_task.set_display_name("Evaluation — rolling backtest (WAPE + MAPE)")
    evaluation_task.after(training_task)
    evaluation_task.set_caching_options(enable_caching=False)
    evaluation_task.set_cpu_limit("4")
    evaluation_task.set_memory_limit("16G")

    # ── Step 5: Champion vs Challenger gate ───────────────────────────────────
    champ_task = champion_vs_challenger_op(
        direction=direction,
        processed_data=preprocess_task.outputs["processed_data"],
        challenger_model=training_task.outputs["model"],
        approval_decision_path=evaluation_task.outputs["approval_decision"],
        project_id=project_id,
        region=region,
        model_display_name=model_display_name,
        champion_label_env=champion_label_env,
        champion_label_role=champion_label_role,
        delta_rel=champ_delta_rel,
        delta_abs=champ_delta_abs,
        forecast_horizon=forecast_horizon,
        backtest_step_days=backtest_step_days,
        evaluation_start_date=evaluation_start_date,
    )
    champ_task.set_display_name("Gate — Champion vs Challenger (WAPE)")
    champ_task.after(evaluation_task)
    champ_task.set_caching_options(enable_caching=False)
    champ_task.set_cpu_limit("4")
    champ_task.set_memory_limit("16G")

    # ── Step 6: Refit on 100% of data ─────────────────────────────────────────
    refit_task = refit_op(
        direction=direction,
        processed_data=preprocess_task.outputs["processed_data"],
        candidate_model=training_task.outputs["model"],
        champion_gate_decision_path=champ_task.outputs["champion_gate_decision"],
    )
    refit_task.set_display_name("Refit — final model on 100% data")
    refit_task.after(champ_task)
    refit_task.set_caching_options(enable_caching=False)
    refit_task.set_cpu_limit("4")
    refit_task.set_memory_limit("16G")

    # ── Step 7: Register in Vertex AI Model Registry ──────────────────────────
    registration_task = model_registration_op(
        direction=direction,
        model=refit_task.outputs["refit_model"],
        champion_gate_decision_path=champ_task.outputs["champion_gate_decision"],
        project_id=project_id,
        region=region,
        model_display_name=model_display_name,
        experiment_name=experiment_name,
        champion_label_env=champion_label_env,
        champion_label_role=champion_label_role,
        archived_label_role=archived_label_role,
        serving_container_image_uri=serving_container_image_uri,
    )
    registration_task.set_display_name("Register — Vertex AI Model Registry")
    registration_task.after(refit_task)
    registration_task.set_caching_options(enable_caching=False)
    registration_task.set_cpu_limit("2")
    registration_task.set_memory_limit("8G")
