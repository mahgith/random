"""
EVALUATION COMPONENT
====================
Step 4 — Runs a rolling backtest to measure forecast quality and
compares the 3-layer model against a constant-mean baseline.

WAPE (Weighted Absolute Percentage Error) is the primary metric:
    WAPE = Σ|y - ŷ| / Σ|y|

Why WAPE over MAPE
------------------
MAPE blows up for near-zero days (common in logistics volumes on weekends
or around holidays).  WAPE weights each day by its volume, so high-volume
days dominate the score — exactly the days that matter most for operations.

Backtest procedure
------------------
For each cutoff in [evaluation_start_date … data_end – horizon]:
  1. Compute L1 exponential baseline from history before the cutoff
  2. Look up L2A multipliers from the saved table
  3. Apply saved L2B Prophet model
  4. Apply saved L3 LightGBM correction
  5. Also compute constant-mean prediction (mean of last `lookback_days`)
  6. Record predictions + actuals for the next `forecast_horizon` days

Metrics reported
----------------
Logged per forecast-week bucket (W1 = days 1-7, W2 = 8-14, …):
    wape_W1..W4, mape_W1..W4

Overall:
    mean_wape_model      — 3-layer model WAPE
    mean_mape_model      — 3-layer model MAPE
    mean_wape_baseline   — constant-mean baseline WAPE
    mean_mape_baseline   — constant-mean baseline MAPE
    wape_lift            — (baseline_wape - model_wape) / baseline_wape
    mape_lift            — same for MAPE
    backtest_cutoffs     — number of cutoffs evaluated
"""

from kfp.dsl import component, Input, Output, Dataset, Model, Metrics

_ML_TRAINING_IMAGE = "europe-west3-docker.pkg.dev/prj-ceva-gr-wkfc-adapt-dev/gr-forecast/ml-training:latest"


@component(base_image=_ML_TRAINING_IMAGE)
def evaluate_op(
    processed_data: Input[Dataset],
    model: Input[Model],
    lookback_days: int,
    half_life_days: int,
    forecast_horizon: int,
    backtest_step_days: int,
    evaluation_start_date: str,
    evaluation_metrics: Output[Metrics],
):
    """
    Rolling backtest comparing the 3-layer model vs a constant-mean baseline.
    Logs WAPE/MAPE per forecast-week bucket and overall lift over baseline.
    """
    import json
    import logging
    import os
    import pickle

    import joblib
    import numpy as np
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    # ── Load data and model bundle ────────────────────────────────────────────
    df = pd.read_parquet(processed_data.path + ".parquet")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    logger.info("Evaluation data: %d rows, date range %s – %s",
                len(df), df["ds"].min().date(), df["ds"].max().date())

    with open(os.path.join(model.path, "config.json")) as f:
        cfg = json.load(f)

    mult_tbl   = pd.read_parquet(os.path.join(model.path, "multiplier_table.parquet"))
    mult_lkp   = mult_tbl.set_index(["week_of_year", "day_of_week"])["multiplier"]

    with open(os.path.join(model.path, "prophet_model.pkl"), "rb") as f:
        prophet_mdl = pickle.load(f)

    lgbm_mdl      = joblib.load(os.path.join(model.path, "lgbm_model.joblib"))
    lgbm_features = json.load(open(os.path.join(model.path, "lgbm_features.json")))

    logger.info("Model bundle loaded (training cutoff: %s)", cfg.get("training_cutoff"))

    # ── Helpers ───────────────────────────────────────────────────────────────
    workday_set = set(df["ds"])
    df_idx      = df.set_index("ds")

    def exp_weights(n: int, half_life: int) -> np.ndarray:
        decay = np.log(2) / half_life
        idx   = np.arange(n)
        w     = np.exp(-decay * (n - 1 - idx))
        return w / w.sum()

    def next_n_workdays(cutoff: pd.Timestamp, n: int) -> list:
        day, result = cutoff + pd.Timedelta(days=1), []
        while len(result) < n:
            if day in workday_set:
                result.append(day)
            day += pd.Timedelta(days=1)
        return result

    def predict_model(cutoff: pd.Timestamp, horizon_dates: list) -> np.ndarray:
        history = df[df["ds"] <= cutoff].tail(lookback_days)
        if len(history) < 2:
            return np.full(len(horizon_dates), np.nan)
        l1_val = float(np.dot(history["y"].values,
                              exp_weights(len(history), half_life_days)))

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
        fdf["prophet_ratio"] = (
            fdf["prophet_pred"] / fdf["l2a_pred"].replace(0, np.nan)
        ).clip(0.1, 1.8)
        fdf["y_structural"] = fdf["l2a_pred"] * fdf["prophet_ratio"]

        for win in (10, 20, 30):
            recent = df[df["ds"] <= cutoff]["y"].tail(win)
            fdf[f"rolling_{win}"] = recent.mean() if len(recent) > 0 else l1_val

        log_corr = lgbm_mdl.predict(fdf[lgbm_features].fillna(0).values)
        return fdf["y_structural"].values * np.exp(log_corr)

    def predict_baseline(cutoff: pd.Timestamp, horizon_dates: list) -> np.ndarray:
        """Constant-mean baseline: mean of the last `lookback_days` actuals."""
        history = df[df["ds"] <= cutoff].tail(lookback_days)
        if len(history) < 1:
            return np.full(len(horizon_dates), np.nan)
        constant = float(history["y"].mean())
        return np.full(len(horizon_dates), constant)

    # ── Rolling backtest ──────────────────────────────────────────────────────
    eval_start  = pd.Timestamp(evaluation_start_date)
    all_dates   = df["ds"].sort_values().tolist()
    last_usable = df["ds"].max() - pd.Timedelta(days=forecast_horizon + 10)
    cutoffs     = [d for d in all_dates
                   if d >= eval_start and d <= last_usable][::backtest_step_days]

    if not cutoffs:
        logger.warning(
            "No cutoffs found between %s and %s. "
            "Check evaluation_start_date or extend the data range.",
            evaluation_start_date, last_usable.date(),
        )
        evaluation_metrics.log_metric("backtest_cutoffs", 0)
        return

    logger.info("Running rolling backtest: %d cutoffs", len(cutoffs))

    records_model    = []
    records_baseline = []

    for cutoff in cutoffs:
        horizon_dates = next_n_workdays(cutoff, forecast_horizon)
        if len(horizon_dates) < forecast_horizon:
            continue

        preds_model    = predict_model(cutoff, horizon_dates)
        preds_baseline = predict_baseline(cutoff, horizon_dates)

        for h_idx, hd in enumerate(horizon_dates, start=1):
            if hd not in df_idx.index:
                continue
            actual = float(df_idx.loc[hd, "y"])
            if actual <= 0:
                continue

            pm = preds_model[h_idx - 1]
            pb = preds_baseline[h_idx - 1]

            if not np.isnan(pm):
                records_model.append({
                    "horizon_day":   h_idx,
                    "actual":        actual,
                    "pred":          pm,
                    "abs_error":     abs(pm - actual),
                    "abs_pct_error": abs(pm - actual) / actual,
                })
            if not np.isnan(pb):
                records_baseline.append({
                    "horizon_day":   h_idx,
                    "actual":        actual,
                    "pred":          pb,
                    "abs_error":     abs(pb - actual),
                    "abs_pct_error": abs(pb - actual) / actual,
                })

    if not records_model:
        logger.warning("No backtest records produced — check evaluation_start_date.")
        evaluation_metrics.log_metric("backtest_cutoffs", len(cutoffs))
        return

    res_model    = pd.DataFrame(records_model)
    res_baseline = pd.DataFrame(records_baseline)

    # ── WAPE / MAPE by forecast-week bucket ───────────────────────────────────
    def week_bucket(h: int) -> str:
        if h <= 7:   return "W1"
        if h <= 14:  return "W2"
        if h <= 21:  return "W3"
        return "W4"

    res_model["week_bucket"] = res_model["horizon_day"].apply(week_bucket)
    for bucket, grp in res_model.groupby("week_bucket"):
        wape_b = float(grp["abs_error"].sum() / (grp["actual"].sum() + 1e-9))
        mape_b = float(grp["abs_pct_error"].mean())
        logger.info("Model [%s] — WAPE: %.4f, MAPE: %.4f (n=%d)",
                    bucket, wape_b, mape_b, len(grp))
        evaluation_metrics.log_metric(f"wape_{bucket}", round(wape_b, 4))
        evaluation_metrics.log_metric(f"mape_{bucket}", round(mape_b, 4))

    # ── Overall metrics ───────────────────────────────────────────────────────
    mean_wape_model    = float(res_model["abs_error"].sum()
                               / (res_model["actual"].sum() + 1e-9))
    mean_mape_model    = float(res_model["abs_pct_error"].mean())

    mean_wape_baseline = float(res_baseline["abs_error"].sum()
                               / (res_baseline["actual"].sum() + 1e-9))
    mean_mape_baseline = float(res_baseline["abs_pct_error"].mean())

    wape_lift = (mean_wape_baseline - mean_wape_model) / (mean_wape_baseline + 1e-9)
    mape_lift = (mean_mape_baseline - mean_mape_model) / (mean_mape_baseline + 1e-9)

    logger.info(
        "Overall — Model WAPE: %.4f, MAPE: %.4f | "
        "Baseline WAPE: %.4f, MAPE: %.4f | "
        "WAPE lift: %.1f%%",
        mean_wape_model, mean_mape_model,
        mean_wape_baseline, mean_mape_baseline,
        wape_lift * 100,
    )

    evaluation_metrics.log_metric("mean_wape_model",    round(mean_wape_model,    4))
    evaluation_metrics.log_metric("mean_mape_model",    round(mean_mape_model,    4))
    evaluation_metrics.log_metric("mean_wape_baseline", round(mean_wape_baseline, 4))
    evaluation_metrics.log_metric("mean_mape_baseline", round(mean_mape_baseline, 4))
    evaluation_metrics.log_metric("wape_lift_pct",      round(wape_lift * 100,    2))
    evaluation_metrics.log_metric("mape_lift_pct",      round(mape_lift * 100,    2))
    evaluation_metrics.log_metric("backtest_cutoffs",   len(cutoffs))
    evaluation_metrics.log_metric("backtest_records",   len(res_model))
