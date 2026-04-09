"""
GCS TRAINING PIPELINE
=====================
A self-contained pipeline that reads data from GCS, preprocesses it,
trains the forecasting model, and evaluates it against a constant-mean baseline.

Architecture: L1 (exponential recency baseline) + L2B Prophet + L3 LightGBM.
No BigQuery, no HP grid search, no L2A multiplier table.

Components
----------
1. read_gcs_op       — GCS CSV  →  raw Dataset artifact
2. preprocess_op     — raw      →  daily feature-enriched Dataset artifact
3. train_op          — data     →  Model artifact + training Metrics
4. evaluate_op       — data + model → evaluation Metrics (WAPE/MAPE vs baseline)
"""

import sys
from pathlib import Path

# Make vertex/ importable so component files can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

from kfp import dsl

from components.read_gcs_op   import read_gcs_op
from components.preprocess_op import preprocess_op
from components.train_op      import train_op
from components.evaluate_op   import evaluate_op


@dsl.pipeline(
    name="gcs-train-pipeline",
    description="3-layer forecasting model (L1+L2A+L2B+L3) trained on GCS data with fixed hyperparameters",
)
def gcs_train_pipeline(
    # ── Data ─────────────────────────────────────────────────────────────────
    project_id: str,
    gcs_uri: str,
    date_column: str,
    target_column: str,
    warehouse_id: str = "GENNEVILLIERS",
    # ── L1 params ────────────────────────────────────────────────────────────
    lookback_days: int = 90,
    half_life_days: int = 30,
    # ── L2B Prophet ───────────────────────────────────────────────────────────
    prophet_changepoint_prior_scale: float = 0.1,
    # ── L3 LightGBM ──────────────────────────────────────────────────────────
    lgbm_n_estimators: int = 1200,
    lgbm_learning_rate: float = 0.05,
    lgbm_num_leaves: int = 63,
    # ── Evaluation ────────────────────────────────────────────────────────────
    forecast_horizon: int = 28,
    backtest_step_days: int = 7,
    evaluation_start_date: str = "2023-01-01",
):
    # Step 1 — read data from GCS
    read_task = read_gcs_op(
        project_id=project_id,
        gcs_uri=gcs_uri,
    )

    # Step 2 — preprocess: filter warehouse, apply 5h shift, aggregate daily, add features
    preprocess_task = preprocess_op(
        raw_data=read_task.outputs["raw_data"],
        date_column=date_column,
        target_column=target_column,
        warehouse_id=warehouse_id,
    )

    # Step 3 — train 3-layer model with fixed hyperparameters
    train_task = train_op(
        processed_data=preprocess_task.outputs["processed_data"],
        lookback_days=lookback_days,
        half_life_days=half_life_days,
        prophet_changepoint_prior_scale=prophet_changepoint_prior_scale,
        lgbm_n_estimators=lgbm_n_estimators,
        lgbm_learning_rate=lgbm_learning_rate,
        lgbm_num_leaves=lgbm_num_leaves,
    )

    # Step 4 — evaluate: rolling backtest vs constant-mean baseline
    evaluate_task = evaluate_op(  # noqa: F841
        processed_data=preprocess_task.outputs["processed_data"],
        model=train_task.outputs["model"],
        lookback_days=lookback_days,
        half_life_days=half_life_days,
        forecast_horizon=forecast_horizon,
        backtest_step_days=backtest_step_days,
        evaluation_start_date=evaluation_start_date,
    )
