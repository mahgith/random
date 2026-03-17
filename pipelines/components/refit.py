"""
REFIT COMPONENT
===============
Trains the final production model on 100% of available historical data
using the best hyperparameters found during the tuning phase.

This step is triggered only after both gates pass:
  1. Threshold evaluation gate  (evaluation_op)
  2. Champion vs Challenger gate (champion_vs_challenger_op)

Why refit?
----------
The candidate model from training_op was evaluated against a rolling backtest
window. That evaluation process confirms the hyperparameters are good. The
refit then applies those same winning hyperparameters to the complete dataset
(including the most recent data up to today) so the production model has the
freshest view of recent patterns before it goes live.

Model bundle saved to refit_model.path/:
    config.json              — direction, hyperparams, training metadata
    multiplier_table.parquet — L2A lookup
    prophet_model.pkl        — serialised Prophet model
    lgbm_model.joblib        — LightGBM model
    lgbm_features.json       — ordered feature list
"""

from kfp.v2.dsl import component, Input, Output, Dataset, Model

_FORECASTING_IMAGE = "europe-west1-docker.pkg.dev/your-gcp-project-id/ml-images/forecasting:latest"


@component(base_image=_FORECASTING_IMAGE)
def refit_op(
    direction: str,
    processed_data: Input[Dataset],
    candidate_model: Input[Model],       # carries best hyperparams in metadata
    champion_gate_decision_path: str,    # path to file written by champion_vs_challenger_op
    refit_model: Output[Model] = None,   # type: ignore[assignment]
):
    """
    Re-train the model on full data with the winning hyperparameters.
    Only runs if the champion gate was passed.
    """
    import json
    import os
    import pickle
    import structlog
    import joblib
    import numpy as np
    import pandas as pd
    from prophet import Prophet
    from common.core.logger import get_logger

    logger = get_logger("refit")
    structlog.contextvars.bind_contextvars(direction=direction)

    # ── Check champion gate ───────────────────────────────────────────────────
    with open(champion_gate_decision_path) as f:
        decision = f.read().strip()

    if decision != "approved":
        logger.warning("Champion gate not passed — skipping refit", decision=decision)
        # Write an empty sentinel so downstream components don't crash
        os.makedirs(refit_model.path, exist_ok=True)
        with open(os.path.join(refit_model.path, "_skipped"), "w") as f:
            f.write("champion_gate_not_passed")
        return

    # ── Recover best hyperparameters from candidate model metadata ────────────
    meta = candidate_model.metadata

    def _cast(v: str):
        try:
            return int(v) if "." not in v else float(v)
        except (ValueError, TypeError):
            return v

    prophet_cps = float(meta.get("prophet_changepoint_prior_scale", 0.1))
    lgbm_n      = int(meta.get("lgbm_n_estimators", 1200))
    lgbm_lr     = float(meta.get("lgbm_learning_rate", 0.05))
    lgbm_leaves = int(meta.get("lgbm_num_leaves", 63))

    logger.info(
        "Hyperparameters recovered from candidate metadata",
        prophet_cps=prophet_cps,
        lgbm_n=lgbm_n,
        lgbm_lr=lgbm_lr,
        lgbm_leaves=lgbm_leaves,
    )

    # ── Load the full dataset ─────────────────────────────────────────────────
    df = pd.read_parquet(processed_data.path + ".parquet")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    with open(os.path.join(candidate_model.path, "config.json")) as f:
        cand_cfg = json.load(f)

    lookback_days  = cand_cfg["lookback_days"]
    half_life_days = cand_cfg["half_life_days"]

    logger.info(
        "Full dataset loaded",
        rows=len(df),
        date_min=str(df["ds"].min().date()),
        date_max=str(df["ds"].max().date()),
    )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def exp_weights(n: int, half_life: int) -> np.ndarray:
        decay = np.log(2) / half_life
        idx   = np.arange(n)
        w     = np.exp(-decay * (n - 1 - idx))
        return w / w.sum()

    lgbm_features = [
        "day_of_week", "week_of_year", "month", "iso_year",
        "is_holiday", "is_pre_holiday", "is_post_holiday",
        "rolling_10", "rolling_20", "rolling_30",
        "y_structural",
    ]

    import logging as _logging
    _logging.getLogger("prophet").setLevel(_logging.WARNING)
    _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

    # ── Layer 2A — multiplier table ───────────────────────────────────────────
    logger.info("Fitting L2A multiplier table")
    l1_baselines = np.full(len(df), np.nan)
    for i in range(lookback_days, len(df)):
        window_y        = df["y"].iloc[i - lookback_days: i].values
        l1_baselines[i] = np.dot(window_y, exp_weights(lookback_days, half_life_days))

    df["l1_baseline"] = l1_baselines
    train_l2a = df.dropna(subset=["l1_baseline"]).copy()
    train_l2a["ratio"] = (train_l2a["y"] / train_l2a["l1_baseline"]).clip(0.2, 3.0)
    multiplier_table = (
        train_l2a.groupby(["week_of_year", "day_of_week"])["ratio"]
        .mean().reset_index().rename(columns={"ratio": "multiplier"})
    )
    mult_lkp = multiplier_table.set_index(["week_of_year", "day_of_week"])["multiplier"]

    # ── Layer 2B — Prophet ────────────────────────────────────────────────────
    logger.info("Fitting Prophet on full data", changepoint_prior_scale=prophet_cps)
    prophet_df = df[["ds", "y", "is_holiday", "is_pre_holiday", "is_post_holiday"]].copy()
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=prophet_cps,
    )
    m.add_regressor("is_holiday",      standardize=False)
    m.add_regressor("is_pre_holiday",  standardize=False)
    m.add_regressor("is_post_holiday", standardize=False)
    m.fit(prophet_df)

    # Structural forecast (needed for L3 targets)
    in_sample   = m.predict(prophet_df[["ds", "is_holiday", "is_pre_holiday", "is_post_holiday"]])
    df          = df.merge(
        in_sample[["ds", "yhat"]].rename(columns={"yhat": "prophet_pred"}),
        on="ds", how="left",
    )
    df["l2a_pred"] = df.apply(
        lambda r: mult_lkp.get((int(r["week_of_year"]), int(r["day_of_week"])), 1.0)
        * r["l1_baseline"] if not np.isnan(r["l1_baseline"]) else np.nan, axis=1
    )
    df["prophet_ratio"] = (df["prophet_pred"] / df["l2a_pred"].replace(0, np.nan)).clip(0.1, 1.8)
    df["y_structural"]  = df["l2a_pred"] * df["prophet_ratio"]

    # ── Layer 3 — LightGBM ────────────────────────────────────────────────────
    logger.info("Fitting LightGBM on full data", n_estimators=lgbm_n, lr=lgbm_lr, leaves=lgbm_leaves)
    train_l3 = df.dropna(subset=["y_structural"] + lgbm_features).copy()
    train_l3 = train_l3[train_l3["y_structural"] > 0].copy()
    train_l3["lgbm_target"] = np.log(train_l3["y"] / train_l3["y_structural"].clip(lower=1.0))

    n   = len(train_l3)
    sw  = exp_weights(n, half_life=60)
    X   = train_l3[lgbm_features].values
    y_t = train_l3["lgbm_target"].values

    import lightgbm as lgb
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=lgbm_n,
        learning_rate=lgbm_lr,
        num_leaves=lgbm_leaves,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    lgbm_model.fit(X, y_t, sample_weight=sw)

    # ── Save final model bundle ───────────────────────────────────────────────
    os.makedirs(refit_model.path, exist_ok=True)

    config = {
        "direction":        direction,
        "training_cutoff":  str(df["ds"].max().date()),
        "lookback_days":    lookback_days,
        "half_life_days":   half_life_days,
        "last_baseline":    float(df["y"].tail(lookback_days).mean()),
        "is_refitted":      True,
        "prophet_changepoint_prior_scale": prophet_cps,
        "lgbm_n_estimators":              lgbm_n,
        "lgbm_learning_rate":             lgbm_lr,
        "lgbm_num_leaves":                lgbm_leaves,
    }
    with open(os.path.join(refit_model.path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    multiplier_table.to_parquet(os.path.join(refit_model.path, "multiplier_table.parquet"), index=False)

    with open(os.path.join(refit_model.path, "prophet_model.pkl"), "wb") as f:
        pickle.dump(m, f)

    joblib.dump(lgbm_model, os.path.join(refit_model.path, "lgbm_model.joblib"))

    with open(os.path.join(refit_model.path, "lgbm_features.json"), "w") as f:
        json.dump(lgbm_features, f)

    # Copy metadata from candidate so registration has full context
    for k, v in candidate_model.metadata.items():
        refit_model.metadata[k] = v
    refit_model.metadata["is_refitted"] = "True"

    logger.info("Refit complete — model bundle ready for registration", artifact_dir=refit_model.path)
