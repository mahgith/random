"""
GCS TRAINING PIPELINE
=====================
A self-contained pipeline that reads data from GCS, preprocesses it,
trains the forecasting model, and evaluates it against a constant-mean baseline.

Architecture: L1 (exponential recency baseline) + L2 Prophet + L3 LightGBM.
No BigQuery, no HP grid search, no L2A multiplier table.

Components
----------
1. read_gcs_op       — GCS CSV  →  raw Dataset artifact
2. preprocess_op     — raw      →  daily feature-enriched Dataset artifact
3. split_op          — full data → train_data (before evaluation_start_date)
4. train_op          — train_data → Model artifact + training Metrics
5. evaluate_op       — full data + model → evaluation Metrics (WAPE/MAPE vs baseline)

Note: evaluate_op receives the full dataset (not just the eval portion) so that
it has the historical context (lookback_days) needed to compute L1 baselines at
the first evaluation cutoffs.
"""

import sys
from pathlib import Path

# Make vertex/ importable so component files can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

from kfp import dsl

from components.read_gcs_op   import read_gcs_op
from components.preprocess_op import preprocess_op
from components.split_op      import split_op
from components.train_op      import train_op
from components.evaluate_op   import evaluate_op


@dsl.pipeline(
    name="gcs-train-pipeline",
    description="3-layer forecasting model (L1+L2 Prophet constrained by L1+L3 LightGBM) trained on GCS data",
)
def gcs_train_pipeline(
    # ── Data ─────────────────────────────────────────────────────────────────
    project_id: str,
    gcs_uri: str,
    date_column: str,
    target_column: str,
    warehouse_id: str = "GENNEVILLIERS",
    data_start_date: str = "2023-06-01",
    # ── L1 params ────────────────────────────────────────────────────────────
    lookback_days: int = 90,
    half_life_days: int = 30,
    # ── L2 Prophet (constrained by L1) ──────────────────────────────────────
    prophet_changepoint_prior_scale: float = 0.1,
    clip_min_ratio: float = 0.1,
    clip_max_ratio: float = 1.8,
    # ── L3 LightGBM ──────────────────────────────────────────────────────────
    l3_retrain_step_days: int = 30,
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
        data_start_date=data_start_date,
    )

    # Step 3 — split: training data ends strictly before evaluation_start_date
    split_task = split_op(
        processed_data=preprocess_task.outputs["processed_data"],
        train_end_date=evaluation_start_date,
    )

    # Step 4 — train on pre-eval data only (no leakage)
    train_task = train_op(
        processed_data=split_task.outputs["train_data"],
        lookback_days=lookback_days,
        half_life_days=half_life_days,
        prophet_changepoint_prior_scale=prophet_changepoint_prior_scale,
        clip_min_ratio=clip_min_ratio,
        clip_max_ratio=clip_max_ratio,
        l3_retrain_step_days=l3_retrain_step_days,
        lgbm_n_estimators=lgbm_n_estimators,
        lgbm_learning_rate=lgbm_learning_rate,
        lgbm_num_leaves=lgbm_num_leaves,
    )

    # Step 5 — evaluate: rolling backtest on held-out eval period
    # Full processed_data passed so evaluate_op has pre-eval history for L1 baselines
    evaluate_task = evaluate_op(  # noqa: F841
        processed_data=preprocess_task.outputs["processed_data"],
        model=train_task.outputs["model"],
        forecast_horizon=forecast_horizon,
        backtest_step_days=backtest_step_days,
        evaluation_start_date=evaluation_start_date,
    )
