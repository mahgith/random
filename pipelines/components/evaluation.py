"""
EVALUATION COMPONENT
====================
Runs a rolling backtest against the fitted model bundle and computes
MAPE, WAPE, and MAE by forecast-week bucket.

WAPE is the primary gate metric (Weighted Absolute Percentage Error):
    WAPE = sum|y - ŷ| / sum|y|
It is more stable than MAPE for logistics volume (not blown up by
near-zero days) and naturally weights high-volume days more heavily.

Rolling backtest logic
----------------------
For each cutoff from evaluation_start_date stepping every backtest_step_days:
  1. Compute L1 exponential baseline from actuals before the cutoff
  2. Apply L2A multiplier lookup from the bundle
  3. Apply L2B Prophet predictions
  4. Compute structural forecast (L2 blend)
  5. Apply L3 LightGBM log-residual correction
  6. Compare predictions to actuals over forecast_horizon workdays

Metrics reported by forecast-week bucket:
    W1: horizon days  1-7
    W2: horizon days  8-14
    W3: horizon days 15-21
    W4: horizon days 22-28

Approval gate: mean WAPE ≤ wape_threshold AND mean MAPE ≤ mape_threshold.
"""

from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, Artifact

_FORECASTING_IMAGE = "europe-west1-docker.pkg.dev/your-gcp-project-id/ml-images/forecasting:latest"


@component(base_image=_FORECASTING_IMAGE)
def evaluation_op(
    direction: str,
    processed_data: Input[Dataset],
    model: Input[Model],
    evaluation_start_date: str,
    forecast_horizon: int,
    backtest_step_days: int,
    mape_threshold: float,
    wape_threshold: float,
    evaluation_metrics: Output[Metrics],
    approval_decision: Output[Artifact],
):
    """Run rolling backtest, compute MAPE + WAPE, and decide whether to approve."""
    import json
    import os
    import pickle
    import structlog
    import joblib
    import numpy as np
    import pandas as pd
    from common.core.logger import get_logger

    logger = get_logger("evaluation")
    structlog.contextvars.bind_contextvars(
        direction=direction,
        evaluation_start_date=evaluation_start_date,
        forecast_horizon=forecast_horizon,
    )

    # ── Load data and model bundle ────────────────────────────────────────────
    df = pd.read_parquet(processed_data.path + ".parquet")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    with open(os.path.join(model.path, "config.json")) as f:
        cfg = json.load(f)

    multiplier_table = pd.read_parquet(os.path.join(model.path, "multiplier_table.parquet"))
    mult_lookup      = multiplier_table.set_index(["week_of_year", "day_of_week"])["multiplier"]

    with open(os.path.join(model.path, "prophet_model.pkl"), "rb") as f:
        prophet_model = pickle.load(f)

    lgbm_model    = joblib.load(os.path.join(model.path, "lgbm_model.joblib"))
    lgbm_features = json.load(open(os.path.join(model.path, "lgbm_features.json")))

    lookback_days  = cfg["lookback_days"]
    half_life_days = cfg["half_life_days"]

    logger.info("Model bundle loaded", training_cutoff=cfg.get("training_cutoff"))

    # ── Helpers ───────────────────────────────────────────────────────────────
    workday_set = set(df["ds"])
    df_indexed  = df.set_index("ds")

    def future_workdays(cutoff: pd.Timestamp, n: int):
        day, result = cutoff + pd.Timedelta(days=1), []
        while len(result) < n:
            if day in workday_set:
                result.append(day)
            day += pd.Timedelta(days=1)
        return result

    def exp_weights(length: int, half_life: int) -> np.ndarray:
        decay = np.log(2) / half_life
        idx   = np.arange(length)
        w     = np.exp(-decay * (length - 1 - idx))
        return w / w.sum()

    def predict_from_cutoff(cutoff: pd.Timestamp, horizon_dates: list) -> pd.Series:
        history = df[df["ds"] <= cutoff].tail(lookback_days)
        if len(history) < 2:
            return pd.Series(np.nan, index=horizon_dates)
        l1_val = float(np.dot(history["y"].values, exp_weights(len(history), half_life_days)))

        fdf = df_indexed.reindex(horizon_dates).copy()
        fdf["ds"] = horizon_dates

        def get_mult(row):
            return mult_lookup.get((int(row["week_of_year"]), int(row["day_of_week"])), 1.0)
        fdf["l2a_pred"] = l1_val * fdf.apply(get_mult, axis=1)

        prophet_future = pd.DataFrame({
            "ds":              horizon_dates,
            "is_holiday":      fdf["is_holiday"].fillna(0).values,
            "is_pre_holiday":  fdf["is_pre_holiday"].fillna(0).values,
            "is_post_holiday": fdf["is_post_holiday"].fillna(0).values,
        })
        fdf["prophet_pred"] = prophet_model.predict(prophet_future)["yhat"].values
        fdf["prophet_ratio"] = (
            fdf["prophet_pred"] / fdf["l2a_pred"].replace(0, np.nan)
        ).clip(0.1, 1.8)
        fdf["y_structural"] = fdf["l2a_pred"] * fdf["prophet_ratio"]

        for win in (10, 20, 30):
            recent = df[df["ds"] <= cutoff]["y"].tail(win)
            fdf[f"rolling_{win}"] = recent.mean() if len(recent) > 0 else l1_val

        log_corr = lgbm_model.predict(fdf[lgbm_features].fillna(0).values)
        return pd.Series(fdf["y_structural"].values * np.exp(log_corr), index=horizon_dates)

    # ── Rolling backtest ──────────────────────────────────────────────────────
    eval_start  = pd.Timestamp(evaluation_start_date)
    cutoffs     = [d for d in df["ds"].sort_values().tolist() if d >= eval_start][::backtest_step_days]
    last_usable = df["ds"].max() - pd.Timedelta(days=forecast_horizon + 10)
    cutoffs     = [c for c in cutoffs if c <= last_usable]
    logger.info("Starting rolling backtest", num_cutoffs=len(cutoffs))

    records = []
    for cutoff in cutoffs:
        horizon_dates = future_workdays(cutoff, forecast_horizon)
        if len(horizon_dates) < forecast_horizon:
            continue
        preds = predict_from_cutoff(cutoff, horizon_dates)
        for h_idx, hd in enumerate(horizon_dates, start=1):
            if hd not in df_indexed.index:
                continue
            actual = df_indexed.loc[hd, "y"]
            pred   = preds.get(hd, np.nan)
            if np.isnan(pred) or actual <= 0:
                continue
            records.append({
                "horizon_day": h_idx,
                "actual":      actual,
                "pred":        pred,
                "abs_error":   abs(pred - actual),
                "abs_pct_error": abs(pred - actual) / actual,
            })

    if not records:
        logger.warning("No backtest records — check evaluation_start_date")
        with open(approval_decision.path, "w") as f:
            f.write("approved")
        return

    results = pd.DataFrame(records)

    # ── WAPE (primary) + MAPE by week bucket ─────────────────────────────────
    def week_bucket(h: int) -> str:
        if h <= 7:   return "W1"
        if h <= 14:  return "W2"
        if h <= 21:  return "W3"
        return "W4"

    results["week_bucket"] = results["horizon_day"].apply(week_bucket)
    for bucket, grp in results.groupby("week_bucket"):
        wape_b = float(grp["abs_error"].sum() / (grp["actual"].sum() + 1e-9))
        mape_b = float(grp["abs_pct_error"].mean())
        logger.info("Bucket", bucket=bucket, wape=round(wape_b, 4), mape=round(mape_b, 4), n=len(grp))
        evaluation_metrics.log_metric(f"wape_{bucket}", round(wape_b, 4))
        evaluation_metrics.log_metric(f"mape_{bucket}", round(mape_b, 4))
        evaluation_metrics.log_metric(f"mae_{bucket}",  round(float(grp["abs_error"].mean()), 2))

    mean_wape = float(results["abs_error"].sum() / (results["actual"].sum() + 1e-9))
    mean_mape = float(results["abs_pct_error"].mean())
    mean_mae  = float(results["abs_error"].mean())

    logger.info(
        "Overall backtest metrics",
        mean_wape=round(mean_wape, 4),
        mean_mape=round(mean_mape, 4),
        mean_mae=round(mean_mae, 1),
    )
    evaluation_metrics.log_metric("mean_wape",        round(mean_wape, 4))
    evaluation_metrics.log_metric("mean_mape",        round(mean_mape, 4))
    evaluation_metrics.log_metric("mean_mae",         round(mean_mae, 2))
    evaluation_metrics.log_metric("wape_threshold",   wape_threshold)
    evaluation_metrics.log_metric("mape_threshold",   mape_threshold)
    evaluation_metrics.log_metric("backtest_cutoffs", len(cutoffs))

    # ── Approval decision — WAPE is the primary gate ──────────────────────────
    passed   = mean_wape <= wape_threshold and mean_mape <= mape_threshold
    decision = "approved" if passed else "rejected"
    logger.info("Approval decision", decision=decision, mean_wape=round(mean_wape, 4), mean_mape=round(mean_mape, 4))

    with open(approval_decision.path, "w") as f:
        f.write(decision)
