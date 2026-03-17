"""
TRAINING COMPONENT
==================
Fits the full 3-layer forecasting model on all available processed data.
Matches the production cells of modeling_inbound.ipynb / modeling_outbound.ipynb.

Architecture
------------
Layer 1 — Exponential recency-weighted baseline
    Computes a weighted mean of recent actuals at inference time.
    No fitting required; only the configuration is stored.
    lookback_days workdays of history, exponential decay with half_life_days.

Layer 2A — Seasonal multiplier table
    For each (week_of_year, day_of_week) pair, stores the mean historical
    ratio of actual / L1-baseline (clipped to [0.2, 3.0]).

Layer 2B — Prophet time-series model
    Yearly seasonality, multiplicative mode.
    Holiday regressors: is_holiday, is_pre_holiday, is_post_holiday.

Layer 3 — LightGBM residual correction
    Target:   log(y_true / y_structural)
    Features: calendar + rolling stats + y_structural
    Weights:  exponential decay (60-day half-life, from end of training data)

Structural forecast (L2, used as L3 input):
    y_struct = L2A_pred * clip(prophet_pred / L2A_pred, 0.1, 1.8)

Model bundle (saved to model.path/):
    config.json              — direction, hyperparams, training metadata
    multiplier_table.parquet — L2A lookup (week_of_year × day_of_week)
    prophet_model.pkl        — serialised Prophet model
    lgbm_model.joblib        — LightGBM model
    lgbm_features.json       — ordered feature list for L3 inference
"""

from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics


@component(
    packages_to_install=[
        "pandas>=2.0.0",
        "pyarrow",
        "numpy>=1.24.0",
        "prophet>=1.1.5",
        "lightgbm>=4.0.0",
        "joblib",
        "holidays>=0.46",
    ],
    base_image="python:3.10-slim",
)
def training_op(
    direction: str,
    processed_data: Input[Dataset],
    # L1
    lookback_days: int,
    half_life_days: int,
    # L2B
    prophet_changepoint_prior_scale: float,
    # L3
    lgbm_n_estimators: int,
    lgbm_learning_rate: float,
    lgbm_num_leaves: int,
    # Outputs
    model: Output[Model] = None,        # type: ignore[assignment]
    metrics: Output[Metrics] = None,    # type: ignore[assignment]
):
    """Fit the 3-layer forecasting model and save the artifact bundle."""
    import json
    import logging
    import os
    import pickle

    import joblib
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from prophet import Prophet

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # ── Load processed data ───────────────────────────────────────────────────
    df = pd.read_parquet(processed_data.path + ".parquet")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    log.info(f"[{direction}] Training data: {len(df)} workdays, "
             f"{df['ds'].min().date()} – {df['ds'].max().date()}")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def exp_weights(n: int, half_life: int) -> np.ndarray:
        """Exponential decay weights, most-recent last."""
        decay = np.log(2) / half_life
        idx = np.arange(n)           # 0 = oldest, n-1 = most recent
        w = np.exp(-decay * (n - 1 - idx))
        return w / w.sum()

    # ── Layer 1 — baseline computation (used to compute L2A ratios) ───────────
    # For each date t, L1(t) = weighted mean of y over the preceding
    # lookback_days workdays.  We compute this across the full training set
    # to build the L2A multiplier table.
    log.info(f"[{direction}] Computing L1 baselines for L2A fitting ...")

    l1_baselines = np.full(len(df), np.nan)
    for i in range(lookback_days, len(df)):
        window_y = df["y"].iloc[i - lookback_days: i].values
        w = exp_weights(lookback_days, half_life_days)
        l1_baselines[i] = np.dot(window_y, w)

    df["l1_baseline"] = l1_baselines

    # ── Layer 2A — multiplier table ───────────────────────────────────────────
    log.info(f"[{direction}] Fitting L2A multiplier table ...")

    train_l2a = df.dropna(subset=["l1_baseline"]).copy()
    # Clip ratios to [0.2, 3.0] to remove outliers
    train_l2a["ratio"] = (train_l2a["y"] / train_l2a["l1_baseline"]).clip(0.2, 3.0)

    multiplier_table = (
        train_l2a.groupby(["week_of_year", "day_of_week"])["ratio"]
        .mean()
        .reset_index()
        .rename(columns={"ratio": "multiplier"})
    )
    log.info(f"[{direction}] L2A table: {len(multiplier_table)} (week, dow) pairs")

    # ── Layer 2B — Prophet ────────────────────────────────────────────────────
    log.info(f"[{direction}] Fitting Prophet ...")

    prophet_df = df[["ds", "y", "is_holiday", "is_pre_holiday", "is_post_holiday"]].copy()
    prophet_df = prophet_df.rename(columns={"y": "y"})  # Prophet expects 'y' and 'ds'

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,   # we model week-of-year via L2A
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=prophet_changepoint_prior_scale,
    )
    m.add_regressor("is_holiday",     standardize=False)
    m.add_regressor("is_pre_holiday", standardize=False)
    m.add_regressor("is_post_holiday", standardize=False)

    # Suppress Prophet's noisy stdout during fitting
    import logging as _logging
    _logging.getLogger("prophet").setLevel(_logging.WARNING)
    _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

    m.fit(prophet_df)

    # In-sample Prophet predictions (needed to compute structural forecast for L3)
    prophet_in_sample = m.predict(prophet_df[["ds", "is_holiday", "is_pre_holiday", "is_post_holiday"]])
    df = df.merge(
        prophet_in_sample[["ds", "yhat"]].rename(columns={"yhat": "prophet_pred"}),
        on="ds", how="left",
    )

    # ── Structural forecast (L2 blend) ────────────────────────────────────────
    # L2A prediction: L1_baseline × multiplier[week_of_year, day_of_week]
    mult_lookup = multiplier_table.set_index(["week_of_year", "day_of_week"])["multiplier"]

    def get_multiplier(row):
        key = (row["week_of_year"], row["day_of_week"])
        return mult_lookup.get(key, 1.0)  # default to 1.0 if unseen

    df["l2a_pred"] = df.apply(get_multiplier, axis=1) * df["l1_baseline"]

    # Clip Prophet/L2A ratio to [0.1, 1.8] — matches notebook approach
    df["prophet_ratio"] = (df["prophet_pred"] / df["l2a_pred"].replace(0, np.nan)).clip(0.1, 1.8)
    df["y_structural"] = df["l2a_pred"] * df["prophet_ratio"]

    # ── Layer 3 — LightGBM residual correction ────────────────────────────────
    log.info(f"[{direction}] Fitting LightGBM residual model ...")

    lgbm_features = [
        "day_of_week", "week_of_year", "month", "iso_year",
        "is_holiday", "is_pre_holiday", "is_post_holiday",
        "rolling_10", "rolling_20", "rolling_30",
        "y_structural",
    ]

    # Need valid structural forecast rows (requires lookback window)
    train_l3 = df.dropna(subset=["y_structural"] + lgbm_features).copy()
    train_l3 = train_l3[train_l3["y_structural"] > 0].copy()

    # Target: log ratio of actual to structural forecast
    train_l3["lgbm_target"] = np.log(
        train_l3["y"] / train_l3["y_structural"].clip(lower=1.0)
    )

    # Sample weights: exponential decay, most-recent data gets higher weight
    n = len(train_l3)
    sample_weights = exp_weights(n, half_life=60)

    X = train_l3[lgbm_features].values
    y_lgbm = train_l3["lgbm_target"].values

    lgbm_model = lgb.LGBMRegressor(
        n_estimators=lgbm_n_estimators,
        learning_rate=lgbm_learning_rate,
        num_leaves=lgbm_num_leaves,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    lgbm_model.fit(X, y_lgbm, sample_weight=sample_weights)

    # In-sample MAPE (on the L3 training set, structural vs. final)
    y_struct_train = train_l3["y_structural"].values
    y_final_train = y_struct_train * np.exp(lgbm_model.predict(X))
    y_true_train  = train_l3["y"].values

    mape_structural = float(np.mean(np.abs(y_true_train - y_struct_train) / (y_true_train + 1e-9)))
    mape_final      = float(np.mean(np.abs(y_true_train - y_final_train)  / (y_true_train + 1e-9)))
    log.info(f"[{direction}] In-sample MAPE — structural: {mape_structural:.3f}  final: {mape_final:.3f}")

    # ── Save model bundle ─────────────────────────────────────────────────────
    os.makedirs(model.path, exist_ok=True)

    # config.json
    config = {
        "direction": direction,
        "training_cutoff": str(df["ds"].max().date()),
        "lookback_days": lookback_days,
        "half_life_days": half_life_days,
        "prophet_changepoint_prior_scale": prophet_changepoint_prior_scale,
        "lgbm_n_estimators": lgbm_n_estimators,
        "lgbm_learning_rate": lgbm_learning_rate,
        "lgbm_num_leaves": lgbm_num_leaves,
    }
    with open(os.path.join(model.path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # L2A multiplier table
    multiplier_table.to_parquet(
        os.path.join(model.path, "multiplier_table.parquet"), index=False
    )

    # L2B Prophet
    with open(os.path.join(model.path, "prophet_model.pkl"), "wb") as f:
        pickle.dump(m, f)

    # L3 LightGBM
    joblib.dump(lgbm_model, os.path.join(model.path, "lgbm_model.joblib"))

    with open(os.path.join(model.path, "lgbm_features.json"), "w") as f:
        json.dump(lgbm_features, f)

    log.info(f"[{direction}] Model bundle saved to {model.path}")

    # ── Log metrics ───────────────────────────────────────────────────────────
    metrics.log_metric("insample_mape_structural", round(mape_structural, 4))
    metrics.log_metric("insample_mape_final",      round(mape_final, 4))
    metrics.log_metric("training_rows",            len(df))
    metrics.log_metric("l3_training_rows",         len(train_l3))

    log.info(f"[{direction}] Training complete.")
