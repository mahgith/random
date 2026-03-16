"""
TRAINING COMPONENT
==================
Responsibility: Train the forecasting model on preprocessed data.

What goes here (from your notebook):
  - Your model instantiation and .fit() call
  - Hyperparameter handling
  - Saving the trained model artifact (joblib/pickle)
  - Logging metrics to Vertex AI Experiments (optional but very useful)

Input:  train_dataset  — processed training data from preprocessing_op
Output: model          — the saved model artifact (uploaded to GCS by KFP)
        metrics        — training metrics (dict serialised as JSON)

NOTE ON THE MODEL ARTIFACT
--------------------------
KFP's Output[Model] gives you a local path. Save your model file there
(e.g. with joblib.dump). KFP uploads it to GCS. The model_registration_op
will receive this GCS path and register it in Vertex AI Model Registry.
"""

from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics


@component(
    packages_to_install=[
        "pandas",
        "pyarrow",
        "scikit-learn",
        "numpy",
        "joblib",
        "google-cloud-aiplatform",  # for Vertex AI Experiments logging
    ],
    base_image="python:3.10-slim",
)
def training_op(
    # ── Inputs ────────────────────────────────────────────────────────────────
    train_dataset: Input[Dataset],
    project_id: str,
    experiment_name: str,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,

    # ── Outputs ───────────────────────────────────────────────────────────────
    model: Output[Model] = None,       # type: ignore[assignment]
    metrics: Output[Metrics] = None,   # type: ignore[assignment]
):
    """Train the forecasting model and save the artifact."""
    import pandas as pd
    import numpy as np
    import joblib
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # ── Load training data ────────────────────────────────────────────────────
    df = pd.read_parquet(train_dataset.path + ".parquet")
    log.info(f"Training data shape: {df.shape}")

    # ── Define features and target ────────────────────────────────────────────
    # List your feature columns here. Adjust to match your preprocessing output.
    target_col = "value"
    exclude_cols = ["date", target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_train = df[feature_cols].values
    y_train = df[target_col].values

    log.info(f"Features ({len(feature_cols)}): {feature_cols}")
    log.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

    # ── Train model ───────────────────────────────────────────────────────────
    # Replace this with your actual model.
    # Common choices for forecasting:
    #   - LightGBM / XGBoost (good default, fast, handles lags well)
    #   - RandomForest (simple baseline)
    #   - Prophet / NeuralProphet (if you want dedicated time-series model)
    #   - LSTM / Transformer (if deep learning)
    #
    # Example with GradientBoostingRegressor (swap in your real model):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    clf = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
    )

    log.info("Training model ...")
    clf.fit(X_train, y_train)

    # ── Training metrics ──────────────────────────────────────────────────────
    y_pred_train = clf.predict(X_train)
    train_mae = float(mean_absolute_error(y_train, y_pred_train))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    log.info(f"Train MAE: {train_mae:.4f} | Train RMSE: {train_rmse:.4f}")

    # Log metrics to KFP Metrics artifact (visible in Vertex AI UI)
    metrics.log_metric("train_mae", train_mae)
    metrics.log_metric("train_rmse", train_rmse)
    metrics.log_metric("n_estimators", n_estimators)
    metrics.log_metric("max_depth", max_depth)

    # ── Save model artifact ───────────────────────────────────────────────────
    # model.path is a local directory. Save your model file(s) inside it.
    import os
    os.makedirs(model.path, exist_ok=True)
    model_file = os.path.join(model.path, "model.joblib")
    joblib.dump(clf, model_file)

    # Save feature list alongside the model — you'll need this at serving time
    feature_file = os.path.join(model.path, "features.json")
    with open(feature_file, "w") as f:
        json.dump(feature_cols, f)

    log.info(f"Model saved to {model_file}")
    log.info("Training complete.")
