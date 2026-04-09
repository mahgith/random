"""
EVALUATION COMPONENT
====================
Step 4 — Rolling backtest comparing the 3-layer model against a
constant-mean baseline. Writes a debug log to GCS for visibility.
"""

from kfp.dsl import component, Input, Output, Dataset, Model, Metrics

_ML_TRAINING_IMAGE = "europe-west3-docker.pkg.dev/prj-ceva-gr-wkfc-adapt-dev/gr-forecast/ml-training:latest"


@component(base_image=_ML_TRAINING_IMAGE)
def evaluate_op(
    processed_data: Input[Dataset],
    model: Input[Model],
    forecast_horizon: int,
    backtest_step_days: int,
    evaluation_start_date: str,
    evaluation_metrics: Output[Metrics],
):
    """Rolling backtest comparing L1+Prophet+LightGBM vs constant-mean baseline."""
    import io
    import json
    import logging
    import os
    import sys
    import traceback

    _debug_log = []
    _debug_gcs_path = "gs://csb-reg-euw3-forecast-data-dev/debug/evaluate_op_debug.txt"

    def p(msg):
        print(msg, flush=True)
        _debug_log.append(str(msg))

    def flush_debug_to_gcs():
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with fs.open(_debug_gcs_path, "w") as f:
                f.write("\n".join(_debug_log))
        except Exception as e:
            print(f"[debug flush failed: {e}]", flush=True)

    p("=== evaluate_op started ===")

    try:
        import joblib
        import numpy as np
        import pandas as pd
        from prophet.serialize import model_from_json
        p("imports OK")
    except Exception as e:
        p(f"IMPORT ERROR: {e}")
        flush_debug_to_gcs()
        raise

    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    try:
        # ── Load data ─────────────────────────────────────────────────────────
        p(f"loading data from: {processed_data.path}.parquet")
        df = pd.read_parquet(processed_data.path + ".parquet")
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").reset_index(drop=True)
        p(f"loaded {len(df)} rows | date range: {df['ds'].min().date()} – {df['ds'].max().date()}")

        # ── Load model bundle ─────────────────────────────────────────────────
        p(f"loading model bundle from: {model.path}")
        with open(os.path.join(model.path, "config.json")) as f:
            cfg = json.load(f)
        lookback_days  = cfg["lookback_days"]
        half_life_days = cfg["half_life_days"]
        p(f"config: {cfg}")

        with open(os.path.join(model.path, "prophet_model.json")) as f:
            prophet_mdl = model_from_json(f.read())
        p("prophet model loaded")

        lgbm_mdl      = joblib.load(os.path.join(model.path, "lgbm_model.joblib"))
        lgbm_features = json.load(open(os.path.join(model.path, "lgbm_features.json")))
        p(f"lgbm model loaded | features: {lgbm_features}")

        flush_debug_to_gcs()  # checkpoint: data + model loaded

        # ── Helpers ───────────────────────────────────────────────────────────
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

            prophet_in = pd.DataFrame({
                "ds":              horizon_dates,
                "is_holiday":      fdf["is_holiday"].fillna(0).values,
                "is_pre_holiday":  fdf["is_pre_holiday"].fillna(0).values,
                "is_post_holiday": fdf["is_post_holiday"].fillna(0).values,
            })
            fdf["y_structural"] = prophet_mdl.predict(prophet_in)["yhat"].values
            fdf["l1_baseline"]  = l1_val

            for win in (10, 20, 30):
                recent = df[df["ds"] <= cutoff]["y"].tail(win)
                fdf[f"rolling_{win}"] = recent.mean() if len(recent) > 0 else l1_val

            log_corr = lgbm_mdl.predict(fdf[lgbm_features].fillna(0).values)
            return fdf["y_structural"].values * np.exp(log_corr)

        def predict_baseline(cutoff: pd.Timestamp, n: int) -> np.ndarray:
            history = df[df["ds"] <= cutoff].tail(lookback_days)
            if len(history) < 1:
                return np.full(n, np.nan)
            return np.full(n, float(history["y"].mean()))

        # ── Rolling backtest ──────────────────────────────────────────────────
        eval_start  = pd.Timestamp(evaluation_start_date)
        all_dates   = df["ds"].sort_values().tolist()
        last_usable = df["ds"].max() - pd.Timedelta(days=forecast_horizon + 10)
        cutoffs     = [d for d in all_dates
                       if d >= eval_start and d <= last_usable][::backtest_step_days]

        p(f"backtest: {len(cutoffs)} cutoffs from {evaluation_start_date} every {backtest_step_days} days")

        if not cutoffs:
            p("WARNING: no cutoffs found — check evaluation_start_date")
            evaluation_metrics.log_metric("backtest_cutoffs", 0)
            flush_debug_to_gcs()
            return

        records_model    = []
        records_baseline = []

        for i, cutoff in enumerate(cutoffs):
            horizon_dates = next_n_workdays(cutoff, forecast_horizon)
            if len(horizon_dates) < forecast_horizon:
                continue

            preds_model    = predict_model(cutoff, horizon_dates)
            preds_baseline = predict_baseline(cutoff, forecast_horizon)

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

        p(f"backtest done: {len(records_model)} model records, {len(records_baseline)} baseline records")
        flush_debug_to_gcs()  # checkpoint: backtest done

        if not records_model:
            p("WARNING: no backtest records — check evaluation_start_date")
            evaluation_metrics.log_metric("backtest_cutoffs", len(cutoffs))
            flush_debug_to_gcs()
            return

        res_model    = pd.DataFrame(records_model)
        res_baseline = pd.DataFrame(records_baseline)

        # ── WAPE / MAPE by forecast-week bucket ───────────────────────────────
        def week_bucket(h: int) -> str:
            if h <= 7:   return "W1"
            if h <= 14:  return "W2"
            if h <= 21:  return "W3"
            return "W4"

        res_model["week_bucket"] = res_model["horizon_day"].apply(week_bucket)
        for bucket, grp in res_model.groupby("week_bucket"):
            wape_b = float(grp["abs_error"].sum() / (grp["actual"].sum() + 1e-9))
            mape_b = float(grp["abs_pct_error"].mean())
            p(f"[{bucket}] WAPE={wape_b:.4f}  MAPE={mape_b:.4f}  n={len(grp)}")
            evaluation_metrics.log_metric(f"wape_{bucket}", round(wape_b, 4))
            evaluation_metrics.log_metric(f"mape_{bucket}", round(mape_b, 4))

        # ── Overall metrics ───────────────────────────────────────────────────
        mean_wape_model    = float(res_model["abs_error"].sum()
                                   / (res_model["actual"].sum() + 1e-9))
        mean_mape_model    = float(res_model["abs_pct_error"].mean())
        mean_wape_baseline = float(res_baseline["abs_error"].sum()
                                   / (res_baseline["actual"].sum() + 1e-9))
        mean_mape_baseline = float(res_baseline["abs_pct_error"].mean())
        wape_lift = (mean_wape_baseline - mean_wape_model) / (mean_wape_baseline + 1e-9)
        mape_lift = (mean_mape_baseline - mean_mape_model) / (mean_mape_baseline + 1e-9)

        p(f"model   WAPE={mean_wape_model:.4f}  MAPE={mean_mape_model:.4f}")
        p(f"baseline WAPE={mean_wape_baseline:.4f}  MAPE={mean_mape_baseline:.4f}")
        p(f"WAPE lift={wape_lift*100:.1f}%")

        evaluation_metrics.log_metric("mean_wape_model",    round(mean_wape_model,    4))
        evaluation_metrics.log_metric("mean_mape_model",    round(mean_mape_model,    4))
        evaluation_metrics.log_metric("mean_wape_baseline", round(mean_wape_baseline, 4))
        evaluation_metrics.log_metric("mean_mape_baseline", round(mean_mape_baseline, 4))
        evaluation_metrics.log_metric("wape_lift_pct",      round(wape_lift * 100,    2))
        evaluation_metrics.log_metric("mape_lift_pct",      round(mape_lift * 100,    2))
        evaluation_metrics.log_metric("backtest_cutoffs",   len(cutoffs))
        evaluation_metrics.log_metric("backtest_records",   len(res_model))

        p("=== evaluate_op complete ===")
        flush_debug_to_gcs()

    except Exception as e:
        p(f"EVALUATION ERROR: {e}")
        tb_buf = io.StringIO()
        traceback.print_exc(file=tb_buf)
        _debug_log.append(tb_buf.getvalue())
        flush_debug_to_gcs()
        raise
