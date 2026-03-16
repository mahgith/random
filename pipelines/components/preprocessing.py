"""
PREPROCESSING COMPONENT
=======================
Responsibility: Transform raw data into model-ready features.

What goes here (from your notebook):
  - Feature engineering
  - Handling missing values
  - Creating lag features, rolling windows, calendar features (common in forecasting)
  - Train/test split
  - Saving a fitted scaler/encoder if needed (save it as an artifact too)

Input:  raw_dataset   — the GCS artifact written by data_ingestion_op
Output: train_dataset, test_dataset — split and feature-engineered data
"""

from kfp.v2.dsl import component, Input, Output, Dataset


@component(
    packages_to_install=[
        "pandas",
        "pyarrow",
        "scikit-learn",
        "numpy",
    ],
    base_image="python:3.10-slim",
)
def preprocessing_op(
    # ── Inputs ────────────────────────────────────────────────────────────────
    raw_dataset: Input[Dataset],          # comes from data_ingestion_op
    forecast_horizon: int = 30,
    test_size_fraction: float = 0.2,

    # ── Outputs ───────────────────────────────────────────────────────────────
    train_dataset: Output[Dataset] = None,    # type: ignore[assignment]
    test_dataset: Output[Dataset] = None,     # type: ignore[assignment]
):
    """Engineer features and split into train/test sets."""
    import pandas as pd
    import numpy as np
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # ── Load raw data ─────────────────────────────────────────────────────────
    df = pd.read_parquet(raw_dataset.path + ".parquet")
    log.info(f"Loaded raw data: {df.shape}")

    # ── Feature engineering ───────────────────────────────────────────────────
    # This is where you paste your notebook's feature engineering code.
    # Below is a generic forecasting example — replace with your logic.

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # --- Calendar features (common for forecasting) ---
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # --- Lag features ---
    # Rule of thumb: lags of 1, 7, 14, 30 days are a good starting point
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df["value"].shift(lag)

    # --- Rolling statistics ---
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = df["value"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["value"].shift(1).rolling(window).std()

    # Drop rows with NaN from lag features
    df = df.dropna().reset_index(drop=True)
    log.info(f"After feature engineering: {df.shape}, columns: {list(df.columns)}")

    # ── Train / Test split ────────────────────────────────────────────────────
    # For time series: NEVER shuffle. Split chronologically.
    split_idx = int(len(df) * (1 - test_size_fraction))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    log.info(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    # ── Write outputs ─────────────────────────────────────────────────────────
    train_df.to_parquet(train_dataset.path + ".parquet", index=False)
    test_df.to_parquet(test_dataset.path + ".parquet", index=False)

    log.info("Preprocessing complete.")
