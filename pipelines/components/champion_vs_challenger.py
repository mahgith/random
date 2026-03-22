"""
CHAMPION VS CHALLENGER COMPONENT
=================================
Compares the newly trained candidate model (Challenger) against the current
production model (Champion) using the same rolling backtest evaluation.

The Champion is fetched from Vertex AI Model Registry by looking for the
model version labelled role=champion for the given display name and environment.
Both models are evaluated on the same processed_data rolling backtest window,
and WAPE is used as the primary comparison metric.

Gate rule:
    challenger_wape ≤ champion_wape - max(delta_abs, delta_rel × champion_wape)

In plain English: the challenger must improve WAPE by a meaningful margin —
either an absolute 0.5 pp or a relative 3% improvement, whichever is larger.
This prevents noisy marginal improvements from replacing the production model.

If no Champion exists yet (first pipeline run), the gate passes automatically
so the new model can bootstrap the registry.
"""

from kfp.dsl import component, Input, Output, Model, Metrics, Artifact

_FORECASTING_IMAGE = "europe-west1-docker.pkg.dev/your-gcp-project-id/ml-images/forecasting:latest"


@component(base_image=_FORECASTING_IMAGE)
def champion_vs_challenger_op(
    direction: str,
    project_id: str,
    location: str,
    bq_processed_dataset: str,
    bq_processed_table: str,
    challenger_model: Input[Model],
    approval_decision: Input[Artifact],  # written by evaluation_op
    region: str,
    model_display_name: str,
    champion_label_env: str,
    champion_label_role: str,
    delta_rel: float,
    delta_abs: float,
    forecast_horizon: int,
    backtest_step_days: int,
    evaluation_start_date: str,
    champion_gate_metrics: Output[Metrics],
    champion_gate_decision: Output[Artifact],
):
    """Compare challenger vs champion and write approved/rejected to champion_gate_decision."""
    import json
    import os
    import pickle
    import tempfile
    import structlog
    import joblib
    import numpy as np
    import pandas as pd
    from google.cloud import aiplatform, bigquery, storage
    from common.core.logger import get_logger

    logger = get_logger("champion-vs-challenger")
    structlog.contextvars.bind_contextvars(
        direction=direction,
        project_id=project_id,
        model_display_name=model_display_name,
    )

    def _write_decision(decision: str) -> None:
        with open(champion_gate_decision.path, "w") as f:
            f.write(decision)

    # ── Check upstream evaluation gate ────────────────────────────────────────
    with open(approval_decision.path) as f:
        upstream_decision = f.read().strip()

    if upstream_decision != "approved":
        logger.warning("Upstream evaluation gate not passed — skipping champion comparison")
        _write_decision("rejected")
        return

    # ── Load processed data from BigQuery ─────────────────────────────────────
    bq_client = bigquery.Client(project=project_id, location=location)
    processed_table = f"`{project_id}.{bq_processed_dataset}.{bq_processed_table}`"
    df = bq_client.query(f"SELECT * FROM {processed_table} ORDER BY ds").to_dataframe()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # ── Shared evaluation helpers ─────────────────────────────────────────────
    workday_set = set(df["ds"])
    df_indexed  = df.set_index("ds")

    def exp_weights(n: int, half_life: int) -> np.ndarray:
        decay = np.log(2) / half_life
        idx   = np.arange(n)
        w     = np.exp(-decay * (n - 1 - idx))
        return w / w.sum()

    def run_backtest(
        cfg: dict,
        mult_lkp: pd.Series,
        prophet_mdl,
        lgbm_mdl,
        lgbm_features: list,
        label: str,
    ) -> float:
        """Run rolling backtest for one model bundle. Returns overall WAPE."""
        lb = cfg["lookback_days"]
        hl = cfg["half_life_days"]

        eval_start  = pd.Timestamp(evaluation_start_date)
        cutoffs     = [d for d in df["ds"].sort_values().tolist() if d >= eval_start][::backtest_step_days]
        last_usable = df["ds"].max() - pd.Timedelta(days=forecast_horizon + 10)
        cutoffs     = [c for c in cutoffs if c <= last_usable]

        all_actual, all_pred = [], []

        for cutoff in cutoffs:
            history = df[df["ds"] <= cutoff].tail(lb)
            if len(history) < 2:
                continue
            l1_val = float(np.dot(history["y"].values, exp_weights(len(history), hl)))

            day, horizon_dates = cutoff + pd.Timedelta(days=1), []
            while len(horizon_dates) < forecast_horizon:
                if day in workday_set:
                    horizon_dates.append(day)
                day += pd.Timedelta(days=1)
            if len(horizon_dates) < forecast_horizon:
                continue

            fdf = df_indexed.reindex(horizon_dates).copy()
            fdf["ds"] = horizon_dates
            fdf["l2a_pred"] = l1_val * fdf.apply(
                lambda r: mult_lkp.get((int(r["week_of_year"]), int(r["day_of_week"])), 1.0), axis=1
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
                recent = df[df["ds"] <= cutoff]["y"].tail(win)
                fdf[f"rolling_{win}"] = recent.mean() if len(recent) > 0 else l1_val

            log_corr = lgbm_mdl.predict(fdf[lgbm_features].fillna(0).values)
            y_pred   = fdf["y_structural"].values * np.exp(log_corr)

            for hd, yp in zip(horizon_dates, y_pred):
                if hd in df_indexed.index and df_indexed.loc[hd, "y"] > 0:
                    all_actual.append(df_indexed.loc[hd, "y"])
                    all_pred.append(yp)

        if not all_actual:
            return float("inf")

        wape = float(np.sum(np.abs(np.array(all_actual) - np.array(all_pred)))
                     / (np.sum(np.abs(all_actual)) + 1e-9))
        logger.info("Backtest complete", model=label, wape=round(wape, 4), cutoffs=len(cutoffs))
        return wape

    # ── Evaluate Challenger ───────────────────────────────────────────────────
    with open(os.path.join(challenger_model.path, "config.json")) as f:
        chal_cfg = json.load(f)

    chal_mult  = pd.read_parquet(os.path.join(challenger_model.path, "multiplier_table.parquet"))
    chal_mult_lkp = chal_mult.set_index(["week_of_year", "day_of_week"])["multiplier"]

    with open(os.path.join(challenger_model.path, "prophet_model.pkl"), "rb") as f:
        chal_prophet = pickle.load(f)

    chal_lgbm     = joblib.load(os.path.join(challenger_model.path, "lgbm_model.joblib"))
    chal_features = json.load(open(os.path.join(challenger_model.path, "lgbm_features.json")))

    challenger_wape = run_backtest(chal_cfg, chal_mult_lkp, chal_prophet, chal_lgbm, chal_features, "challenger")

    # ── Fetch and evaluate Champion from Vertex AI Registry ──────────────────
    logger.info("Looking up champion model in Vertex AI registry")
    aiplatform.init(project=project_id, location=region)

    existing = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}" AND labels.role="{champion_label_role}" AND labels.env="{champion_label_env}"'
    )

    if not existing:
        logger.info("No champion found — first run, gate passes automatically")
        champion_wape = float("inf")
    else:
        # Pick the most recently created champion
        champion_vertex = sorted(existing, key=lambda m: m.create_time, reverse=True)[0]
        champion_gcs_uri = champion_vertex.uri
        logger.info("Champion found", resource_name=champion_vertex.resource_name, uri=champion_gcs_uri)

        # Download champion bundle from GCS into a temp directory
        storage_client = storage.Client(project=project_id)
        bucket_name = champion_gcs_uri.replace("gs://", "").split("/")[0]
        prefix      = "/".join(champion_gcs_uri.replace("gs://", "").split("/")[1:])
        bucket      = storage_client.bucket(bucket_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            for blob in storage_client.list_blobs(bucket_name, prefix=prefix):
                rel_path  = os.path.relpath(blob.name, prefix)
                local_path = os.path.join(tmpdir, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                blob.download_to_filename(local_path)

            with open(os.path.join(tmpdir, "config.json")) as f:
                champ_cfg = json.load(f)

            champ_mult = pd.read_parquet(os.path.join(tmpdir, "multiplier_table.parquet"))
            champ_mult_lkp = champ_mult.set_index(["week_of_year", "day_of_week"])["multiplier"]

            with open(os.path.join(tmpdir, "prophet_model.pkl"), "rb") as f:
                champ_prophet = pickle.load(f)

            champ_lgbm     = joblib.load(os.path.join(tmpdir, "lgbm_model.joblib"))
            champ_features = json.load(open(os.path.join(tmpdir, "lgbm_features.json")))

            champion_wape = run_backtest(
                champ_cfg, champ_mult_lkp, champ_prophet, champ_lgbm, champ_features, "champion"
            )

    # ── Gate decision ─────────────────────────────────────────────────────────
    required_improvement = max(delta_abs, delta_rel * champion_wape) if champion_wape != float("inf") else 0.0
    wape_target          = champion_wape - required_improvement
    wins                 = (challenger_wape <= wape_target) or (champion_wape == float("inf"))

    champion_gate_metrics.log_metric("champion_wape",        round(champion_wape, 4) if champion_wape != float("inf") else -1.0)
    champion_gate_metrics.log_metric("challenger_wape",      round(challenger_wape, 4))
    champion_gate_metrics.log_metric("required_improvement", round(required_improvement, 4))
    champion_gate_metrics.log_metric("wape_target",          round(wape_target, 4) if champion_wape != float("inf") else -1.0)
    champion_gate_metrics.log_metric("wins",                 1.0 if wins else 0.0)

    decision = "approved" if wins else "rejected"
    logger.info(
        "Champion vs Challenger decision",
        decision=decision,
        champion_wape=round(champion_wape, 4) if champion_wape != float("inf") else "N/A",
        challenger_wape=round(challenger_wape, 4),
        required_improvement=round(required_improvement, 4),
    )

    _write_decision(decision)
