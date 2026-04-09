"""
TRAINING COMPONENT
==================
Step 3 — Fits the 3-layer forecasting model with fixed hyperparameters
(no grid search).

Architecture
------------
Layer 1  — Exponential recency-weighted baseline (smooth level anchor)
Layer 2  — Prophet (yearly seasonality, multiplicative) constrained by L1
           via ratio clipping: y_structural = L1 * clip(prophet / L1, min, max)
Layer 3  — LightGBM log-residual correction on remaining ratio error

Model bundle saved to model.path/:
    config.json              — hyperparameters + training metadata
    prophet_model.json       — serialised Prophet model (JSON, not pickle)
    lgbm_model.joblib        — LightGBM model
    lgbm_features.json       — ordered feature list for L3 inference
"""

from kfp.dsl import component, Input, Output, Dataset, Model, Metrics

_ML_TRAINING_IMAGE = "europe-west3-docker.pkg.dev/prj-ceva-gr-wkfc-adapt-dev/gr-forecast/ml-training:latest"


@component(base_image=_ML_TRAINING_IMAGE)
def train_op(
    processed_data: Input[Dataset],
    lookback_days: int,
    half_life_days: int,
    prophet_changepoint_prior_scale: float,
    clip_min_ratio: float,
    clip_max_ratio: float,
    lgbm_n_estimators: int,
    lgbm_learning_rate: float,
    lgbm_num_leaves: int,
    model: Output[Model],
    metrics: Output[Metrics],
):
    """Fit L1 + L2 (Prophet constrained by L1) + L3 LightGBM on the full dataset."""
    import json
    import os
    import sys
    import traceback

    # Write debug log to GCS so we can always find it regardless of Cloud Logging
    _debug_log = []
    _debug_gcs_path = "gs://csb-reg-euw3-forecast-data-dev/debug/train_op_debug.txt"

    def p(msg):
        print(msg, flush=True)
        _debug_log.append(msg)

    def flush_debug_to_gcs():
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with fs.open(_debug_gcs_path, "w") as f:
                f.write("\n".join(_debug_log))
        except Exception as e:
            print(f"[debug flush failed: {e}]", flush=True)

    p("=== train_op started ===")

    # ── Imports — logged one by one so we know exactly which one fails ────────
    try:
        p("importing joblib...")
        import joblib
        p("importing lightgbm...")
        import lightgbm as lgb
        p("importing numpy...")
        import numpy as np
        p("importing pandas...")
        import pandas as pd
        p("importing prophet...")
        from prophet import Prophet
        from prophet.serialize import model_to_json
        p("all imports OK")
    except Exception as e:
        p(f"IMPORT ERROR: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise

    # ── Helpers ───────────────────────────────────────────────────────────────
    def exp_weights(n: int, half_life: int) -> "np.ndarray":
        decay = np.log(2) / half_life
        idx   = np.arange(n)
        w     = np.exp(-decay * (n - 1 - idx))
        return w / w.sum()

    def compute_l1(series_y: "pd.Series", lb: int, hl: int) -> "np.ndarray":
        baselines = np.full(len(series_y), np.nan)
        vals = series_y.values
        for i in range(lb, len(vals)):
            baselines[i] = float(np.dot(vals[i - lb: i], exp_weights(lb, hl)))
        return baselines

    def wape(y_true: "np.ndarray", y_pred: "np.ndarray") -> float:
        return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-9))

    def mape(y_true: "np.ndarray", y_pred: "np.ndarray") -> float:
        return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-9)))

    # ── Main training logic — wrapped so any error surfaces in logs ───────────
    try:
        lgbm_features = [
            "day_of_week", "week_of_year", "month", "iso_year",
            "is_holiday", "is_pre_holiday", "is_post_holiday",
            "rolling_10", "rolling_20", "rolling_30",
            "l1_baseline",
            "prophet_yhat",
            "raw_ratio",
            "clipped_ratio",
            "y_structural",
        ]

        # Load data
        p(f"loading data from: {processed_data.path}.parquet")
        df = pd.read_parquet(processed_data.path + ".parquet")
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").reset_index(drop=True)
        p(f"loaded {len(df)} rows | columns: {list(df.columns)}")
        p(f"date range: {df['ds'].min().date()} – {df['ds'].max().date()}")

        # L1 baseline
        p("computing L1 baseline...")
        df["l1_baseline"] = compute_l1(df["y"], lookback_days, half_life_days)
        p(f"L1 done. valid rows: {int(df['l1_baseline'].notna().sum())} / {len(df)}")

        # L2B Prophet
        p("fitting Prophet...")
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
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
        p("Prophet fitted")

        in_sample = prophet_mdl.predict(
            prophet_df[["ds", "is_holiday", "is_pre_holiday", "is_post_holiday"]]
        )
        df["prophet_yhat"] = in_sample["yhat"].values
        p(f"prophet_yhat: min={df['prophet_yhat'].min():.2f}  max={df['prophet_yhat'].max():.2f}")

        # L2 structural forecast: constrain Prophet via ratio clipping against L1
        eps = 1e-9
        df["raw_ratio"]     = df["prophet_yhat"] / (df["l1_baseline"] + eps)
        df["clipped_ratio"] = df["raw_ratio"].clip(lower=clip_min_ratio, upper=clip_max_ratio)
        df["y_structural"]  = df["l1_baseline"] * df["clipped_ratio"]
        p(f"y_structural (L1 * clipped ratio): min={df['y_structural'].min():.2f}  max={df['y_structural'].max():.2f}")
        p(f"ratio clipping [{clip_min_ratio}, {clip_max_ratio}]: "
          f"clipped {int((df['raw_ratio'] != df['clipped_ratio']).sum())} / {int(df['raw_ratio'].notna().sum())} rows")

        flush_debug_to_gcs()  # checkpoint: prophet done

        # L3 LightGBM
        p("fitting LightGBM...")
        train_l3 = df.dropna(subset=lgbm_features).copy()
        train_l3 = train_l3[(train_l3["y_structural"] > 0) & (train_l3["y"] > 0)].copy()
        train_l3["lgbm_target"] = np.log(
            train_l3["y"] / (train_l3["y_structural"] + eps)
        )
        p(f"LightGBM training rows: {len(train_l3)}")

        n  = len(train_l3)
        X  = train_l3[lgbm_features].values
        yl = train_l3["lgbm_target"].values
        sw = exp_weights(n, half_life=60)

        # Data diagnostics before fit
        p(f"X shape: {X.shape}  dtype: {X.dtype}")
        p(f"y shape: {yl.shape}  dtype: {yl.dtype}")
        p(f"NaN in X: {int(np.isnan(X).sum())}  Inf in X: {int(np.isinf(X).sum())}")
        p(f"NaN in y: {int(np.isnan(yl).sum())}  Inf in y: {int(np.isinf(yl).sum())}")

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
        lgbm_mdl.fit(X, yl, sample_weight=sw)
        p("LightGBM fitted")

        # In-sample diagnostics
        valid = df.dropna(subset=lgbm_features).copy()
        valid = valid[valid["y_structural"] > 0].copy()
        if len(valid) > 0:
            y_pred_in = valid["y_structural"].values * np.exp(
                lgbm_mdl.predict(valid[lgbm_features].values)
            )
            iw = wape(valid["y"].values, y_pred_in)
            im = mape(valid["y"].values, y_pred_in)
            p(f"in-sample WAPE: {iw:.4f}  MAPE: {im:.4f}")
            metrics.log_metric("insample_wape", round(iw, 4))
            metrics.log_metric("insample_mape", round(im, 4))
        else:
            p("WARNING: no valid rows for in-sample diagnostics")

        metrics.log_metric("training_rows",                   len(df))
        metrics.log_metric("prophet_changepoint_prior_scale", prophet_changepoint_prior_scale)
        metrics.log_metric("lgbm_n_estimators",               lgbm_n_estimators)
        metrics.log_metric("lgbm_learning_rate",               lgbm_learning_rate)
        metrics.log_metric("lgbm_num_leaves",                  lgbm_num_leaves)

        # Save model bundle
        p(f"saving model bundle to: {model.path}")
        os.makedirs(model.path, exist_ok=True)

        config = {
            "training_cutoff":                 str(df["ds"].max().date()),
            "training_rows":                   len(df),
            "lookback_days":                   lookback_days,
            "half_life_days":                  half_life_days,
            "prophet_changepoint_prior_scale": prophet_changepoint_prior_scale,
            "clip_min_ratio":                  clip_min_ratio,
            "clip_max_ratio":                  clip_max_ratio,
            "lgbm_n_estimators":               lgbm_n_estimators,
            "lgbm_learning_rate":              lgbm_learning_rate,
            "lgbm_num_leaves":                 lgbm_num_leaves,
        }
        with open(os.path.join(model.path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        p("saved config.json")

        with open(os.path.join(model.path, "prophet_model.json"), "w") as f:
            f.write(model_to_json(prophet_mdl))
        p("saved prophet_model.json")

        joblib.dump(lgbm_mdl, os.path.join(model.path, "lgbm_model.joblib"))
        p("saved lgbm_model.joblib")

        with open(os.path.join(model.path, "lgbm_features.json"), "w") as f:
            json.dump(lgbm_features, f)
        p("saved lgbm_features.json")

        model.metadata.update({k: str(v) for k, v in config.items()})
        p("=== train_op complete ===")
        flush_debug_to_gcs()

    except Exception as e:
        p(f"TRAINING ERROR: {e}")
        traceback.print_exc(file=sys.stdout)
        _debug_log.append(f"TRAINING ERROR: {e}")
        import io
        tb_buf = io.StringIO()
        traceback.print_exc(file=tb_buf)
        _debug_log.append(tb_buf.getvalue())
        flush_debug_to_gcs()
        raise
