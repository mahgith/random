"""
EVALUATION COMPONENT
====================
Responsibility: Evaluate the trained model on the held-out test set
                and decide whether it meets the quality bar to be registered.

What goes here (from your notebook):
  - Your evaluation metrics code
  - Comparison to a baseline or previously deployed model (optional)
  - A pass/fail decision (used to gate model registration)

Output: evaluation_metrics — metrics artifact (visible in Vertex AI UI)
        approval_decision  — "approved" or "rejected" (gates registration)

WHY GATE ON APPROVAL?
----------------------
In a production pipeline you don't want every run to overwrite your
deployed model. The approval_decision output lets the pipeline skip
registration if the model doesn't beat the threshold.
"""

from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics, ClassificationMetrics


@component(
    packages_to_install=[
        "pandas",
        "pyarrow",
        "scikit-learn",
        "numpy",
        "joblib",
    ],
    base_image="python:3.10-slim",
)
def evaluation_op(
    # ── Inputs ────────────────────────────────────────────────────────────────
    model: Input[Model],
    test_dataset: Input[Dataset],
    mae_threshold: float = 10.0,   # reject if MAE > this value; tune to your problem

    # ── Outputs ───────────────────────────────────────────────────────────────
    evaluation_metrics: Output[Metrics] = None,   # type: ignore[assignment]
    approval_decision: Output[str] = None,          # type: ignore[assignment]  "approved"/"rejected"
):
    """Evaluate model on test set and decide if it should be registered."""
    import pandas as pd
    import numpy as np
    import joblib
    import json
    import os
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # ── Load model ────────────────────────────────────────────────────────────
    model_file = os.path.join(model.path, "model.joblib")
    clf = joblib.load(model_file)

    feature_file = os.path.join(model.path, "features.json")
    with open(feature_file) as f:
        feature_cols = json.load(f)

    # ── Load test data ────────────────────────────────────────────────────────
    df = pd.read_parquet(test_dataset.path + ".parquet")
    log.info(f"Test data shape: {df.shape}")

    target_col = "value"
    X_test = df[feature_cols].values
    y_test = df[target_col].values

    # ── Predict ───────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)

    # ── Compute metrics ───────────────────────────────────────────────────────
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    test_mae = float(mean_absolute_error(y_test, y_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    # MAPE — useful for forecasting; skip if y values can be 0
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100)

    log.info(f"Test MAE:  {test_mae:.4f}")
    log.info(f"Test RMSE: {test_rmse:.4f}")
    log.info(f"Test MAPE: {mape:.2f}%")

    # Log to Vertex AI UI
    evaluation_metrics.log_metric("test_mae", test_mae)
    evaluation_metrics.log_metric("test_rmse", test_rmse)
    evaluation_metrics.log_metric("test_mape_pct", mape)
    evaluation_metrics.log_metric("mae_threshold", mae_threshold)

    # ── Approval decision ─────────────────────────────────────────────────────
    # Write "approved" or "rejected" to the output.
    # The pipeline will check this before running model_registration_op.
    decision = "approved" if test_mae <= mae_threshold else "rejected"
    log.info(f"Approval decision: {decision} (MAE={test_mae:.4f}, threshold={mae_threshold})")

    # KFP Output[str] — write to .path as a plain text file
    with open(approval_decision.path, "w") as f:
        f.write(decision)
