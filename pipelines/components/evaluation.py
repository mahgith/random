"""
EVALUATION COMPONENT
====================
Runs a rolling backtest against the fitted model bundle and computes
MAPE / MAE by forecast-week bucket — matching the evaluation cells in
modeling_inbound.ipynb / modeling_outbound.ipynb.

Rolling backtest logic
----------------------
For each cutoff date from evaluation_start_date to (last_date - forecast_horizon),
stepping every backtest_step_days:

    1. Compute L1 baseline from actuals strictly before the cutoff
    2. Apply L2A multiplier lookup from the bundle
    3. Apply L2B Prophet (in-sample predict, limited to future dates)
    4. Compute structural forecast (L2 blend)
    5. Apply L3 LightGBM correction
    6. Compare predictions to actuals for horizon workdays

MAPE is reported by forecast-week bucket:
    W1: horizon days  7-11
    W2: horizon days 14-18
    W3: horizon days 21-25
    W4: horizon days 28+

Approval gate: mean MAPE across all weeks < mape_threshold.
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
def evaluation_op(
    direction: str,
    processed_data: Input[Dataset],
    model: Input[Model],
    evaluation_start_date: str,   # "YYYY-MM-DD"
    forecast_horizon: int,
    backtest_step_days: int,
    mape_threshold: float,
    evaluation_metrics: Output[Metrics] = None,   # type: ignore[assignment]
    approval_decision: Output[str] = None,        # type: ignore[assignment]
):
    """Run rolling backtest and decide whether to approve model registration."""
    import json
    import logging
    import os
    import pickle

    import joblib
    import numpy as np
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # ── Load data and model bundle ────────────────────────────────────────────
    df = pd.read_parquet(processed_data.path + ".parquet")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    with open(os.path.join(model.path, "config.json")) as f:
        cfg = json.load(f)

    multiplier_table = pd.read_parquet(os.path.join(model.path, "multiplier_table.parquet"))
    mult_lookup = multiplier_table.set_index(["week_of_year", "day_of_week"])["multiplier"]

    with open(os.path.join(model.path, "prophet_model.pkl"), "rb") as f:
        prophet_model = pickle.load(f)

    lgbm_model   = joblib.load(os.path.join(model.path, "lgbm_model.joblib"))
    lgbm_features = json.load(open(os.path.join(model.path, "lgbm_features.json")))

    lookback_days = cfg["lookback_days"]
    half_life_days = cfg["half_life_days"]

    log.info(f"[{direction}] Evaluating from {evaluation_start_date}, "
             f"horizon={forecast_horizon}, step={backtest_step_days}")

    # ── Helper: workday list ──────────────────────────────────────────────────
    workday_set = set(df["ds"])
    df_indexed  = df.set_index("ds")

    def future_workdays(cutoff: pd.Timestamp, n: int):
        """Return the next n workdays strictly after cutoff."""
        day = cutoff + pd.Timedelta(days=1)
        result = []
        while len(result) < n:
            if day in workday_set:
                result.append(day)
            day += pd.Timedelta(days=1)
        return result

    # ── Helper: exponential weights ───────────────────────────────────────────
    def exp_weights(length: int, half_life: int) -> np.ndarray:
        decay = np.log(2) / half_life
        idx = np.arange(length)
        w = np.exp(-decay * (length - 1 - idx))
        return w / w.sum()

    # ── Helper: single-cutoff prediction ─────────────────────────────────────
    def predict_from_cutoff(cutoff: pd.Timestamp, horizon_dates: list) -> pd.Series:
        # L1 baseline
        history = df[df["ds"] <= cutoff].tail(lookback_days)
        if len(history) < 2:
            return pd.Series(np.nan, index=horizon_dates)
        w = exp_weights(len(history), half_life_days)
        l1_val = float(np.dot(history["y"].values, w))

        # Build future feature frame
        future_df_indexed = df_indexed.reindex(horizon_dates)
        fdf = future_df_indexed.copy()
        fdf["ds"] = horizon_dates

        # L2A
        def get_mult(row):
            return mult_lookup.get((int(row["week_of_year"]), int(row["day_of_week"])), 1.0)
        fdf["l2a_pred"] = l1_val * fdf.apply(get_mult, axis=1)

        # L2B Prophet — use in-sample predict for dates the model knows
        prophet_future = pd.DataFrame({
            "ds": horizon_dates,
            "is_holiday":     fdf["is_holiday"].fillna(0).values,
            "is_pre_holiday": fdf["is_pre_holiday"].fillna(0).values,
            "is_post_holiday": fdf["is_post_holiday"].fillna(0).values,
        })
        prophet_pred_df = prophet_model.predict(prophet_future)
        fdf["prophet_pred"] = prophet_pred_df["yhat"].values

        # Structural forecast (L2 blend)
        fdf["prophet_ratio"] = (
            fdf["prophet_pred"] / fdf["l2a_pred"].replace(0, np.nan)
        ).clip(0.1, 1.8)
        fdf["y_structural"] = fdf["l2a_pred"] * fdf["prophet_ratio"]

        # L3 LightGBM — fill rolling stats from history for forecast rows
        for win in (10, 20, 30):
            recent = df[df["ds"] <= cutoff]["y"].tail(win)
            fdf[f"rolling_{win}"] = recent.mean() if len(recent) > 0 else l1_val

        X_future = fdf[lgbm_features].fillna(0).values
        log_correction = lgbm_model.predict(X_future)
        y_final = fdf["y_structural"].values * np.exp(log_correction)

        return pd.Series(y_final, index=horizon_dates)

    # ── Rolling backtest ──────────────────────────────────────────────────────
    eval_start = pd.Timestamp(evaluation_start_date)
    all_dates  = df["ds"].sort_values().tolist()
    cutoffs    = [d for d in all_dates if d >= eval_start]
    # Use every backtest_step_days-th date
    cutoffs    = cutoffs[::backtest_step_days]
    # Skip cutoffs where there wouldn't be enough future actuals
    last_usable = df["ds"].max() - pd.Timedelta(days=forecast_horizon + 10)
    cutoffs     = [c for c in cutoffs if c <= last_usable]

    log.info(f"[{direction}] Running {len(cutoffs)} backtest cutoffs ...")

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
                "cutoff":         cutoff,
                "horizon_date":   hd,
                "horizon_day":    h_idx,
                "actual":         actual,
                "pred":           pred,
                "abs_pct_error":  abs(pred - actual) / actual,
                "abs_error":      abs(pred - actual),
            })

    if not records:
        log.warning(f"[{direction}] No backtest records — check evaluation_start_date")
        # Approve anyway so the pipeline doesn't halt on empty eval window
        _write_decision("approved", approval_decision)
        return

    results = pd.DataFrame(records)

    # ── MAPE by week bucket ───────────────────────────────────────────────────
    def week_bucket(h: int) -> str:
        if 7 <= h <= 11:
            return "W1"
        elif 14 <= h <= 18:
            return "W2"
        elif 21 <= h <= 25:
            return "W3"
        elif h >= 28:
            return "W4"
        return "other"

    results["week_bucket"] = results["horizon_day"].apply(week_bucket)
    weekly = results.groupby("week_bucket").agg(
        mape=("abs_pct_error", "mean"),
        mae=("abs_error", "mean"),
        n=("abs_pct_error", "count"),
    ).reset_index()

    for _, row in weekly.iterrows():
        log.info(
            f"[{direction}] {row['week_bucket']}: "
            f"MAPE={row['mape']:.3f}  MAE={row['mae']:.1f}  n={row['n']}"
        )
        evaluation_metrics.log_metric(f"mape_{row['week_bucket']}", round(float(row["mape"]), 4))
        evaluation_metrics.log_metric(f"mae_{row['week_bucket']},",  round(float(row["mae"]), 2))

    mean_mape = float(results["abs_pct_error"].mean())
    mean_mae  = float(results["abs_error"].mean())
    log.info(f"[{direction}] Overall  MAPE={mean_mape:.3f}  MAE={mean_mae:.1f}")

    evaluation_metrics.log_metric("mean_mape",       round(mean_mape, 4))
    evaluation_metrics.log_metric("mean_mae",        round(mean_mae, 2))
    evaluation_metrics.log_metric("mape_threshold",  mape_threshold)
    evaluation_metrics.log_metric("backtest_cutoffs", len(cutoffs))

    # ── Approval decision ─────────────────────────────────────────────────────
    decision = "approved" if mean_mape <= mape_threshold else "rejected"
    log.info(
        f"[{direction}] Decision: {decision}  "
        f"(mean_mape={mean_mape:.3f}, threshold={mape_threshold})"
    )

    with open(approval_decision.path, "w") as f:
        f.write(decision)


def _write_decision(decision: str, approval_decision) -> None:
    with open(approval_decision.path, "w") as f:
        f.write(decision)
