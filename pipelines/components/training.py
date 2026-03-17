"""
TRAINING COMPONENT
==================
Hyperparameter tuning + final candidate model fitting for the 3-layer
forecasting architecture. Each HP combination is tracked as a run in
Vertex AI Experiments so you can compare them in the GCP console.

Architecture
------------
Layer 1  — Exponential recency-weighted baseline (no training; config stored)
Layer 2A — Seasonal multiplier table (week_of_year × day_of_week)
Layer 2B — Prophet time-series model (yearly seasonality, multiplicative)
Layer 3  — LightGBM log-residual correction

Hyperparameter tuning
---------------------
The hp_grid_json parameter defines a grid over:
    prophet_changepoint_prior_scale  (L2B)
    lgbm_n_estimators                (L3)
    lgbm_learning_rate               (L3)
    lgbm_num_leaves                  (L3)

For each combination we run a mini rolling backtest (tuning_num_cutoffs
cutoffs) and compute WAPE. The winning combination is selected by lowest
WAPE. All runs are logged to Vertex AI Experiments.

After tuning, the candidate model is trained on 100% of the data using
the best hyperparameters and saved as the model artifact.

Model bundle saved to model.path/:
    config.json              — direction, best hyperparams, training metadata
    multiplier_table.parquet — L2A lookup (week_of_year × day_of_week)
    prophet_model.pkl        — serialised Prophet model
    lgbm_model.joblib        — LightGBM model
    lgbm_features.json       — ordered feature list for L3 inference
"""

from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics

_FORECASTING_IMAGE = "europe-west1-docker.pkg.dev/your-gcp-project-id/ml-images/forecasting:latest"


@component(base_image=_FORECASTING_IMAGE)
def training_op(
    direction: str,
    processed_data: Input[Dataset],
    # L1 baseline
    lookback_days: int,
    half_life_days: int,
    # Hyperparameter grid (JSON-encoded dict of lists)
    hp_grid_json: str,
    tuning_num_cutoffs: int,
    # Vertex AI Experiments
    project_id: str,
    region: str,
    experiment_name: str,
    run_prefix: str,
    # Outputs
    model: Output[Model] = None,        # type: ignore[assignment]
    metrics: Output[Metrics] = None,    # type: ignore[assignment]
):
    """Tune hyperparameters via mini-backtest, then train the candidate model."""
    import itertools
    import json
    import os
    import pickle
    import structlog
    from datetime import datetime

    import joblib
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from prophet import Prophet
    from google.cloud import aiplatform
    from common.core.logger import get_logger

    logger = get_logger("training")
    structlog.contextvars.bind_contextvars(direction=direction, project_id=project_id)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_parquet(processed_data.path + ".parquet")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    logger.info("Training data loaded", rows=len(df), date_max=str(df["ds"].max().date()))

    # ── Helpers ───────────────────────────────────────────────────────────────
    def exp_weights(n: int, half_life: int) -> np.ndarray:
        decay = np.log(2) / half_life
        idx   = np.arange(n)
        w     = np.exp(-decay * (n - 1 - idx))
        return w / w.sum()

    def compute_l1(df_hist: pd.DataFrame, lb: int, hl: int) -> np.ndarray:
        baselines = np.full(len(df_hist), np.nan)
        for i in range(lb, len(df_hist)):
            window_y = df_hist["y"].iloc[i - lb: i].values
            baselines[i] = np.dot(window_y, exp_weights(lb, hl))
        return baselines

    lgbm_features = [
        "day_of_week", "week_of_year", "month", "iso_year",
        "is_holiday", "is_pre_holiday", "is_post_holiday",
        "rolling_10", "rolling_20", "rolling_30",
        "y_structural",
    ]

    def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-9))

    def train_bundle(
        df_fit: pd.DataFrame,
        prophet_cps: float,
        lgbm_n: int,
        lgbm_lr: float,
        lgbm_leaves: int,
    ) -> tuple:
        """Fit L2A + L2B + L3 on df_fit. Returns (multiplier_table, prophet_model, lgbm_model)."""
        import logging as _logging
        _logging.getLogger("prophet").setLevel(_logging.WARNING)
        _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

        # L1 baselines (for L2A ratio computation)
        df_fit = df_fit.copy()
        df_fit["l1_baseline"] = compute_l1(df_fit, lookback_days, half_life_days)

        # L2A multiplier table
        train_l2a = df_fit.dropna(subset=["l1_baseline"]).copy()
        train_l2a["ratio"] = (train_l2a["y"] / train_l2a["l1_baseline"]).clip(0.2, 3.0)
        mult_tbl = (
            train_l2a.groupby(["week_of_year", "day_of_week"])["ratio"]
            .mean().reset_index().rename(columns={"ratio": "multiplier"})
        )
        mult_lkp = mult_tbl.set_index(["week_of_year", "day_of_week"])["multiplier"]

        # L2B Prophet
        prophet_df = df_fit[["ds", "y", "is_holiday", "is_pre_holiday", "is_post_holiday"]].copy()
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

        # In-sample structural forecast (needed to compute L3 targets)
        in_sample = m.predict(prophet_df[["ds", "is_holiday", "is_pre_holiday", "is_post_holiday"]])
        df_fit = df_fit.merge(
            in_sample[["ds", "yhat"]].rename(columns={"yhat": "prophet_pred"}),
            on="ds", how="left",
        )
        df_fit["l2a_pred"] = df_fit.apply(
            lambda r: mult_lkp.get((int(r["week_of_year"]), int(r["day_of_week"])), 1.0)
            * r["l1_baseline"], axis=1
        )
        df_fit["prophet_ratio"] = (df_fit["prophet_pred"] / df_fit["l2a_pred"].replace(0, np.nan)).clip(0.1, 1.8)
        df_fit["y_structural"]  = df_fit["l2a_pred"] * df_fit["prophet_ratio"]

        # L3 LightGBM
        train_l3 = df_fit.dropna(subset=["y_structural"] + lgbm_features).copy()
        train_l3 = train_l3[train_l3["y_structural"] > 0].copy()
        train_l3["lgbm_target"] = np.log(train_l3["y"] / train_l3["y_structural"].clip(lower=1.0))

        n     = len(train_l3)
        X     = train_l3[lgbm_features].values
        y_lgb = train_l3["lgbm_target"].values
        sw    = exp_weights(n, half_life=60)

        lgbm_mdl = lgb.LGBMRegressor(
            n_estimators=lgbm_n,
            learning_rate=lgbm_lr,
            num_leaves=lgbm_leaves,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        lgbm_mdl.fit(X, y_lgb, sample_weight=sw)

        return mult_tbl, m, lgbm_mdl

    def mini_backtest_wape(
        df_eval: pd.DataFrame,
        mult_tbl: pd.DataFrame,
        prophet_mdl,
        lgbm_mdl,
        num_cutoffs: int,
    ) -> float:
        """Run a mini rolling backtest and return overall WAPE."""
        mult_lkp = mult_tbl.set_index(["week_of_year", "day_of_week"])["multiplier"]
        workday_set = set(df_eval["ds"])
        df_idx      = df_eval.set_index("ds")

        all_dates  = df_eval["ds"].sort_values().tolist()
        step       = max(1, len(all_dates) // (num_cutoffs + 1))
        cutoffs    = all_dates[lookback_days::step][:num_cutoffs]

        all_actual, all_pred = [], []
        for cutoff in cutoffs:
            history = df_eval[df_eval["ds"] <= cutoff].tail(lookback_days)
            if len(history) < 2:
                continue
            l1_val = float(np.dot(history["y"].values, exp_weights(len(history), half_life_days)))

            # Next 28 workdays
            day, horizon_dates = cutoff + pd.Timedelta(days=1), []
            while len(horizon_dates) < 28:
                if day in workday_set:
                    horizon_dates.append(day)
                day += pd.Timedelta(days=1)

            if not horizon_dates:
                continue

            fdf = df_idx.reindex(horizon_dates).copy()
            fdf["ds"] = horizon_dates
            fdf["l2a_pred"] = l1_val * fdf.apply(
                lambda r: mult_lkp.get((int(r["week_of_year"]), int(r["day_of_week"])), 1.0),
                axis=1,
            )
            prophet_in = pd.DataFrame({
                "ds":              horizon_dates,
                "is_holiday":      fdf["is_holiday"].fillna(0).values,
                "is_pre_holiday":  fdf["is_pre_holiday"].fillna(0).values,
                "is_post_holiday": fdf["is_post_holiday"].fillna(0).values,
            })
            fdf["prophet_pred"] = prophet_mdl.predict(prophet_in)["yhat"].values
            fdf["prophet_ratio"] = (fdf["prophet_pred"] / fdf["l2a_pred"].replace(0, np.nan)).clip(0.1, 1.8)
            fdf["y_structural"]  = fdf["l2a_pred"] * fdf["prophet_ratio"]

            for win in (10, 20, 30):
                recent = df_eval[df_eval["ds"] <= cutoff]["y"].tail(win)
                fdf[f"rolling_{win}"] = recent.mean() if len(recent) > 0 else l1_val

            log_corr = lgbm_mdl.predict(fdf[lgbm_features].fillna(0).values)
            y_pred   = fdf["y_structural"].values * np.exp(log_corr)

            for hd, yp in zip(horizon_dates, y_pred):
                if hd in df_idx.index:
                    all_actual.append(df_idx.loc[hd, "y"])
                    all_pred.append(yp)

        if not all_actual:
            return float("inf")
        return wape(np.array(all_actual), np.array(all_pred))

    # ── Hyperparameter grid search ────────────────────────────────────────────
    hp_grid = json.loads(hp_grid_json)

    # Extract each param list; fall back to a single default if not in grid
    cps_list   = hp_grid.get("prophet_changepoint_prior_scale", [0.1])
    n_list     = hp_grid.get("lgbm_n_estimators", [1200])
    lr_list    = hp_grid.get("lgbm_learning_rate", [0.05])
    leaf_list  = hp_grid.get("lgbm_num_leaves", [63])

    combinations = list(itertools.product(cps_list, n_list, lr_list, leaf_list))
    logger.info("Starting HP grid search", num_combinations=len(combinations))

    # Initialise Vertex AI Experiments for this tuning run
    aiplatform.init(
        project=project_id,
        location=region,
        experiment=f"{experiment_name}-{timestamp}",
    )

    best_wape   = float("inf")
    best_params = {}

    for i, (cps, n_est, lr, n_leaves) in enumerate(combinations):
        run_name = f"{run_prefix}-{timestamp}-combo{i}"
        logger.info(
            "Evaluating combination",
            combo=i + 1,
            total=len(combinations),
            prophet_cps=cps,
            lgbm_n=n_est,
            lgbm_lr=lr,
            lgbm_leaves=n_leaves,
        )

        aiplatform.start_run(run=run_name)
        aiplatform.log_params({
            "prophet_changepoint_prior_scale": cps,
            "lgbm_n_estimators":               n_est,
            "lgbm_learning_rate":              lr,
            "lgbm_num_leaves":                 n_leaves,
            "tuning_num_cutoffs":              tuning_num_cutoffs,
        })

        try:
            mult_tbl, prophet_mdl, lgbm_mdl = train_bundle(df, cps, n_est, lr, n_leaves)
            combo_wape = mini_backtest_wape(df, mult_tbl, prophet_mdl, lgbm_mdl, tuning_num_cutoffs)
            aiplatform.log_metrics({"tuning_wape": round(combo_wape, 4)})
            logger.info("Combination evaluated", wape=round(combo_wape, 4))

            if combo_wape < best_wape:
                best_wape   = combo_wape
                best_params = {
                    "prophet_changepoint_prior_scale": cps,
                    "lgbm_n_estimators":               n_est,
                    "lgbm_learning_rate":              lr,
                    "lgbm_num_leaves":                 n_leaves,
                }
        except Exception as e:
            logger.warning("Combination failed — skipping", error=str(e))
            aiplatform.log_metrics({"tuning_wape": -1.0})
        finally:
            aiplatform.end_run()

    if not best_params:
        raise RuntimeError("All HP combinations failed during tuning")

    logger.info("Tuning complete", best_params=best_params, best_tuning_wape=round(best_wape, 4))

    # ── Train candidate model on 100% of data with best hyperparameters ───────
    logger.info("Training candidate model on full dataset with best parameters")
    mult_tbl, prophet_mdl, lgbm_mdl = train_bundle(
        df,
        best_params["prophet_changepoint_prior_scale"],
        best_params["lgbm_n_estimators"],
        best_params["lgbm_learning_rate"],
        best_params["lgbm_num_leaves"],
    )

    # In-sample MAPE / WAPE on full data (diagnostic only — not the gate)
    import logging as _logging
    _logging.getLogger("prophet").setLevel(_logging.WARNING)
    _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

    df2 = df.copy()
    df2["l1_baseline"] = compute_l1(df2, lookback_days, half_life_days)
    mult_lkp2          = mult_tbl.set_index(["week_of_year", "day_of_week"])["multiplier"]
    df2["l2a_pred"]    = df2.apply(
        lambda r: mult_lkp2.get((int(r["week_of_year"]), int(r["day_of_week"])), 1.0)
        * r["l1_baseline"] if not np.isnan(r["l1_baseline"]) else np.nan, axis=1
    )
    prophet_pred_in    = prophet_mdl.predict(
        df2[["ds", "is_holiday", "is_pre_holiday", "is_post_holiday"]]
    )
    df2["prophet_pred"] = prophet_pred_in["yhat"].values
    df2["prophet_ratio"] = (df2["prophet_pred"] / df2["l2a_pred"].replace(0, np.nan)).clip(0.1, 1.8)
    df2["y_structural"]  = df2["l2a_pred"] * df2["prophet_ratio"]

    valid = df2.dropna(subset=["y_structural"] + lgbm_features).copy()
    valid = valid[valid["y_structural"] > 0]
    if len(valid) > 0:
        X_in   = valid[lgbm_features].values
        y_final_in = valid["y_structural"].values * np.exp(lgbm_mdl.predict(X_in))
        insample_wape = wape(valid["y"].values, y_final_in)
        insample_mape = float(np.mean(np.abs(valid["y"].values - y_final_in) / (valid["y"].values + 1e-9)))
        logger.info("In-sample diagnostics", insample_wape=round(insample_wape, 4), insample_mape=round(insample_mape, 4))
        metrics.log_metric("insample_wape", round(insample_wape, 4))
        metrics.log_metric("insample_mape", round(insample_mape, 4))

    # ── Save model bundle ─────────────────────────────────────────────────────
    os.makedirs(model.path, exist_ok=True)

    config = {
        "direction":     direction,
        "training_cutoff": str(df["ds"].max().date()),
        "lookback_days":  lookback_days,
        "half_life_days": half_life_days,
        "last_baseline":  float(df["y"].tail(lookback_days).mean()),
        **best_params,
    }
    with open(os.path.join(model.path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    mult_tbl.to_parquet(os.path.join(model.path, "multiplier_table.parquet"), index=False)

    with open(os.path.join(model.path, "prophet_model.pkl"), "wb") as f:
        pickle.dump(prophet_mdl, f)

    joblib.dump(lgbm_mdl, os.path.join(model.path, "lgbm_model.joblib"))

    with open(os.path.join(model.path, "lgbm_features.json"), "w") as f:
        json.dump(lgbm_features, f)

    # Store best hyperparams in metadata so the refit component can read them
    model.metadata.update({k: str(v) for k, v in best_params.items()})
    model.metadata["tuning_wape"] = str(round(best_wape, 4))
    model.metadata["training_cutoff"] = str(df["ds"].max().date())

    logger.info("Candidate model bundle saved", artifact_dir=model.path)

    metrics.log_metric("tuning_best_wape", round(best_wape, 4))
    metrics.log_metric("training_rows",    len(df))
