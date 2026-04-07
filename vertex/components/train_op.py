"""
TRAINING COMPONENT
==================
Step 3 — Fits the 3-layer forecasting model with fixed hyperparameters
(no grid search).

Architecture
------------
Layer 1  — Exponential recency-weighted baseline
               Weights the last `lookback_days` actuals with exponential
               decay (half-life = `half_life_days`).  Pure formula, no fit.
               Result: l1_baseline (one value per training row).

Layer 2B — Prophet (yearly seasonality, multiplicative)
               Adds French-holiday regressors.  Captures trend + yearly
               seasonality.  Output: y_structural = prophet yhat.
               (Layer 2A / seasonal multiplier table has been removed.)

Layer 3  — LightGBM log-residual correction
               Learns the log-ratio between actuals and the Prophet
               structural forecast: log(y / y_structural).
               Features: calendar fields + rolling means + l1_baseline.
               Final prediction: y_structural × exp(lgbm_correction).

Model bundle saved to model.path/:
    config.json              — hyperparameters + training metadata
    prophet_model.pkl        — serialised Prophet model
    lgbm_model.joblib        — LightGBM model
    lgbm_features.json       — ordered feature list for L3 inference
"""

from kfp.dsl import component, Input, Output, Dataset, Model, Metrics

_ML_TRAINING_IMAGE = "europe-west3-docker.pkg.dev/prj-ceva-gr-wkfc-adapt-dev/gr-forecast/ml-training:latest"


@component(base_image=_ML_TRAINING_IMAGE)
def train_op(
    processed_data: Input[Dataset],
    # L1 params
    lookback_days: int,
    half_life_days: int,
    # L2B Prophet (fixed)
    prophet_changepoint_prior_scale: float,
    # L3 LightGBM (fixed)
    lgbm_n_estimators: int,
    lgbm_learning_rate: float,
    lgbm_num_leaves: int,
    # Outputs
    model: Output[Model],
    metrics: Output[Metrics],
):
    """Fit L1 + L2B Prophet + L3 LightGBM on the full dataset with fixed hyperparameters."""
    import json
    import logging
    import os
    import pickle

    import joblib
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from prophet import Prophet

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    # Features for L3 LightGBM
    # l1_baseline is the recency-weighted mean — a strong short-term signal
    # y_structural is Prophet's yhat — the trend + seasonality signal
    lgbm_features = [
        "day_of_week", "week_of_year", "month", "iso_year",
        "is_holiday", "is_pre_holiday", "is_post_holiday",
        "rolling_10", "rolling_20", "rolling_30",
        "l1_baseline",
        "y_structural",
    ]

    # ── Helpers ───────────────────────────────────────────────────────────────
    def exp_weights(n: int, half_life: int) -> np.ndarray:
        decay = np.log(2) / half_life
        idx   = np.arange(n)
        w     = np.exp(-decay * (n - 1 - idx))
        return w / w.sum()

    def compute_l1(series_y: pd.Series, lb: int, hl: int) -> np.ndarray:
        baselines = np.full(len(series_y), np.nan)
        vals = series_y.values
        for i in range(lb, len(vals)):
            window = vals[i - lb: i]
            baselines[i] = float(np.dot(window, exp_weights(lb, hl)))
        return baselines

    def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-9))

    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-9)))

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_parquet(processed_data.path + ".parquet")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    logger.info("Training data loaded: %d rows, date range %s – %s",
                len(df), df["ds"].min().date(), df["ds"].max().date())

    # ── L1 — Exponential recency-weighted baseline ────────────────────────────
    # Computes a weighted recent mean for every row.
    # Rows before index `lookback_days` will be NaN (not enough history yet).
    df["l1_baseline"] = compute_l1(df["y"], lookback_days, half_life_days)
    logger.info("L1 baseline computed. Valid rows: %d / %d",
                int(df["l1_baseline"].notna().sum()), len(df))

    # ── L2B — Prophet ─────────────────────────────────────────────────────────
    # Prophet is trained on the full series.
    # Its yhat becomes y_structural — the trend + yearly seasonality signal
    # that L3 will then correct.
    prophet_df = df[["ds", "y", "is_holiday", "is_pre_holiday", "is_post_holiday"]].copy()
    prophet_mdl = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=prophet_changepoint_prior_scale,
    )
    prophet_mdl.add_regressor("is_holiday",      standardize=False)
    prophet_mdl.add_regressor("is_pre_holiday",  standardize=False)
    prophet_mdl.add_regressor("is_post_holiday", standardize=False)
    prophet_mdl.fit(prophet_df)
    logger.info("L2B Prophet fitted")

    # In-sample Prophet predictions to build y_structural for each training row
    in_sample = prophet_mdl.predict(
        prophet_df[["ds", "is_holiday", "is_pre_holiday", "is_post_holiday"]]
    )
    df["y_structural"] = in_sample["yhat"].values
    logger.info("y_structural (Prophet yhat) computed. Min: %.2f, Max: %.2f",
                float(df["y_structural"].min()), float(df["y_structural"].max()))

    # ── L3 — LightGBM log-residual correction ─────────────────────────────────
    # Target: log(actual / prophet_structural)
    # The model learns whatever systematic pattern Prophet missed.
    train_l3 = df.dropna(subset=lgbm_features).copy()
    train_l3 = train_l3[train_l3["y_structural"] > 0].copy()
    train_l3["lgbm_target"] = np.log(
        train_l3["y"] / train_l3["y_structural"].clip(lower=1.0)
    )

    n   = len(train_l3)
    X   = train_l3[lgbm_features].values
    y_l = train_l3["lgbm_target"].values
    sw  = exp_weights(n, half_life=60)

    lgbm_mdl = lgb.LGBMRegressor(
        n_estimators=lgbm_n_estimators,
        learning_rate=lgbm_learning_rate,
        num_leaves=lgbm_num_leaves,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    lgbm_mdl.fit(X, y_l, sample_weight=sw)
    logger.info("L3 LightGBM fitted on %d rows", n)

    # ── In-sample diagnostics ────────────────────────────────────────────────
    # These metrics are optimistic (model saw this data) but useful for
    # a quick sanity check.
    valid = df.dropna(subset=lgbm_features).copy()
    valid = valid[valid["y_structural"] > 0].copy()
    if len(valid) > 0:
        X_in      = valid[lgbm_features].values
        y_pred_in = valid["y_structural"].values * np.exp(lgbm_mdl.predict(X_in))
        insample_wape = wape(valid["y"].values, y_pred_in)
        insample_mape = mape(valid["y"].values, y_pred_in)
        logger.info("In-sample diagnostics — WAPE: %.4f, MAPE: %.4f",
                    insample_wape, insample_mape)
        metrics.log_metric("insample_wape", round(insample_wape, 4))
        metrics.log_metric("insample_mape", round(insample_mape, 4))
    else:
        logger.warning("No valid rows for in-sample diagnostics")

    metrics.log_metric("training_rows",                   len(df))
    metrics.log_metric("prophet_changepoint_prior_scale", prophet_changepoint_prior_scale)
    metrics.log_metric("lgbm_n_estimators",               lgbm_n_estimators)
    metrics.log_metric("lgbm_learning_rate",               lgbm_learning_rate)
    metrics.log_metric("lgbm_num_leaves",                  lgbm_num_leaves)

    # ── Save model bundle ─────────────────────────────────────────────────────
    os.makedirs(model.path, exist_ok=True)

    config = {
        "training_cutoff":                 str(df["ds"].max().date()),
        "training_rows":                   len(df),
        "lookback_days":                   lookback_days,
        "half_life_days":                  half_life_days,
        "prophet_changepoint_prior_scale": prophet_changepoint_prior_scale,
        "lgbm_n_estimators":               lgbm_n_estimators,
        "lgbm_learning_rate":              lgbm_learning_rate,
        "lgbm_num_leaves":                 lgbm_num_leaves,
    }
    with open(os.path.join(model.path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(model.path, "prophet_model.pkl"), "wb") as f:
        pickle.dump(prophet_mdl, f)

    joblib.dump(lgbm_mdl, os.path.join(model.path, "lgbm_model.joblib"))

    with open(os.path.join(model.path, "lgbm_features.json"), "w") as f:
        json.dump(lgbm_features, f)

    model.metadata.update({k: str(v) for k, v in config.items()})
    logger.info("Model bundle saved to: %s", model.path)
