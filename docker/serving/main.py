"""
VERTEX AI PREDICTION CONTAINER — 3-Layer Forecasting Model
===========================================================
Serves the package-volume forecasting model bundle via HTTP.

Vertex AI expects two endpoints:
  GET  /health   → {"status": "healthy"}
  POST /predict  → {"predictions": [...]}

Request payload format
----------------------
{
  "instances": [
    {"date": "2025-01-06"},
    {"date": "2025-01-07"},
    ...
  ]
}

Response payload format
-----------------------
{
  "predictions": [
    {"date": "2025-01-06", "yhat": 12345.6},
    ...
  ]
}

Model bundle layout (loaded from AIP_STORAGE_URI at startup)
-------------------------------------------------------------
  config.json              — direction, hyperparams, training cutoff
  multiplier_table.parquet — L2A (week_of_year × day_of_week → multiplier)
  prophet_model.pkl        — serialised Prophet model
  lgbm_model.joblib        — LightGBM residual corrector
  lgbm_features.json       — ordered feature list expected by LightGBM
"""

import json
import os
import pickle
import logging

import joblib
import numpy as np
import pandas as pd
import holidays as hol_lib
from fastapi import FastAPI, Request

from common.core.logger import get_logger

app = FastAPI()
logger = get_logger("serving")

# ── Global model state (loaded once at startup) ───────────────────────────────
_config = None
_multiplier_table = None
_prophet_model = None
_lgbm_model = None
_lgbm_features = None


@app.on_event("startup")
def load_model() -> None:
    """
    Vertex AI mounts the model artifact directory at AIP_STORAGE_URI.
    We load all bundle files into memory here so predictions are fast.
    """
    global _config, _multiplier_table, _prophet_model, _lgbm_model, _lgbm_features

    model_dir = os.environ.get("AIP_STORAGE_URI", "/app/model")
    logger.info("Loading model bundle", model_dir=model_dir)

    with open(os.path.join(model_dir, "config.json")) as f:
        _config = json.load(f)

    _multiplier_table = pd.read_parquet(
        os.path.join(model_dir, "multiplier_table.parquet")
    )

    with open(os.path.join(model_dir, "prophet_model.pkl"), "rb") as f:
        _prophet_model = pickle.load(f)

    _lgbm_model = joblib.load(os.path.join(model_dir, "lgbm_model.joblib"))

    with open(os.path.join(model_dir, "lgbm_features.json")) as f:
        _lgbm_features = json.load(f)

    logger.info(
        "Model bundle loaded",
        direction=_config.get("direction"),
        training_cutoff=_config.get("training_cutoff"),
    )


# ── Health endpoint (required by Vertex AI) ───────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy"}


# ── Prediction endpoint (required by Vertex AI) ───────────────────────────────
@app.post("/predict")
async def predict(request: Request):
    """
    Runs the 3-layer model for each requested date.

    Steps per date:
      L1  Exponential-weighted baseline from config (uses training cutoff stats)
      L2A Multiplier from (week_of_year, day_of_week) lookup table
      L2B Prophet yearly-seasonality trend
      L3  LightGBM log-residual correction
    """
    body = await request.json()
    instances = body.get("instances", [])

    if not instances:
        return {"error": "No data provided in 'instances' key"}

    # Parse dates
    dates = pd.to_datetime([inst.get("date") for inst in instances])
    df = pd.DataFrame({"ds": dates})
    df = df.sort_values("ds").reset_index(drop=True)

    # ── French holiday features ───────────────────────────────────────────────
    years = df["ds"].dt.year.unique().tolist()
    fr_holidays = set()
    for yr in years:
        fr_holidays.update(hol_lib.France(years=yr).keys())

    holiday_dates = pd.to_datetime(sorted(fr_holidays))
    df["is_holiday"] = df["ds"].dt.date.astype("datetime64[ns]").isin(holiday_dates).astype(int)

    # Pre/post holiday (nearest workday before/after each holiday)
    holiday_set = set(holiday_dates.normalize())
    workday_series = df["ds"].tolist()
    pre_holiday_dates, post_holiday_dates = set(), set()
    for h in holiday_set:
        before = [d for d in workday_series if d < h]
        after  = [d for d in workday_series if d > h]
        if before:
            pre_holiday_dates.add(max(before))
        if after:
            post_holiday_dates.add(min(after))
    df["is_pre_holiday"]  = df["ds"].isin(pre_holiday_dates).astype(int)
    df["is_post_holiday"] = df["ds"].isin(post_holiday_dates).astype(int)

    # ── Calendar features ─────────────────────────────────────────────────────
    iso = df["ds"].dt.isocalendar()
    df["iso_year"]     = iso["year"].astype(int)
    df["week_of_year"] = iso["week"].astype(int)
    df["day_of_week"]  = df["ds"].dt.dayofweek
    df["month"]        = df["ds"].dt.month

    # Rolling stats: not available at serving time without history;
    # use the last known training baseline as a constant approximation.
    for win in (10, 20, 30):
        df[f"rolling_{win}"] = _config.get("last_baseline", 10000.0)

    # ── L2A multiplier lookup ─────────────────────────────────────────────────
    mult_lookup = _multiplier_table.set_index(["week_of_year", "day_of_week"])["multiplier"]

    def get_multiplier(row):
        return mult_lookup.get((int(row["week_of_year"]), int(row["day_of_week"])), 1.0)

    l1_baseline = _config.get("last_baseline", 10000.0)
    df["l2a_pred"] = l1_baseline * df.apply(get_multiplier, axis=1)

    # ── L2B Prophet ───────────────────────────────────────────────────────────
    prophet_input = df[["ds", "is_holiday", "is_pre_holiday", "is_post_holiday"]].copy()
    prophet_out   = _prophet_model.predict(prophet_input)
    df["prophet_pred"] = prophet_out["yhat"].values

    # Structural blend (same clip as training)
    df["prophet_ratio"] = (df["prophet_pred"] / df["l2a_pred"].replace(0, np.nan)).clip(0.1, 1.8)
    df["y_structural"]  = df["l2a_pred"] * df["prophet_ratio"]

    # ── L3 LightGBM ───────────────────────────────────────────────────────────
    X = df[_lgbm_features].fillna(0).values
    log_correction = _lgbm_model.predict(X)
    df["yhat"] = df["y_structural"] * np.exp(log_correction)

    # ── Format response ───────────────────────────────────────────────────────
    predictions = [
        {"date": str(row["ds"].date()), "yhat": round(float(row["yhat"]), 2)}
        for _, row in df.iterrows()
    ]

    logger.info("Prediction complete", num_dates=len(predictions))
    return {"predictions": predictions}
