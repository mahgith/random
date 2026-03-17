"""
PREPROCESSING COMPONENT
=======================
Adds calendar features, French public-holiday flags, and rolling
statistics to the raw daily time series produced by data_ingestion_op.

This matches the feature engineering cells in modeling_inbound.ipynb /
modeling_outbound.ipynb.

Input:  raw_dataset   — daily workday series [ds, y]
Output: processed_data — enriched DataFrame with all features used by training
        and evaluation components
"""

from kfp.v2.dsl import component, Input, Output, Dataset


@component(
    packages_to_install=[
        "pandas>=2.0.0",
        "pyarrow",
        "numpy>=1.24.0",
        "holidays>=0.46",
    ],
    base_image="python:3.10-slim",
)
def preprocessing_op(
    direction: str,                # "inbound" or "outbound" — used only for logging
    raw_dataset: Input[Dataset],
    processed_data: Output[Dataset] = None,  # type: ignore[assignment]
):
    """Add calendar, holiday, and rolling features to the daily time series."""
    import logging
    import numpy as np
    import pandas as pd
    import holidays as hol_lib

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_parquet(raw_dataset.path + ".parquet")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    log.info(f"[{direction}] Loaded {len(df)} workdays")

    # ── French public holidays ────────────────────────────────────────────────
    years = df["ds"].dt.year.unique().tolist()
    fr_holidays = set()
    for yr in years:
        fr_holidays.update(hol_lib.France(years=yr).keys())

    holiday_dates = pd.to_datetime(sorted(fr_holidays))

    df["is_holiday"] = df["ds"].dt.date.astype("datetime64[ns]").isin(
        holiday_dates
    ).astype(int)

    # Pre/post holiday: workday immediately before/after a holiday
    holiday_set = set(holiday_dates.normalize())
    pre_holiday_dates  = set()
    post_holiday_dates = set()

    workday_series = df["ds"].tolist()
    workday_index  = {d: i for i, d in enumerate(workday_series)}

    for h in holiday_set:
        # Find the nearest workday before h
        candidates_before = [d for d in workday_series if d < h]
        if candidates_before:
            pre_holiday_dates.add(max(candidates_before))
        # Find the nearest workday after h
        candidates_after = [d for d in workday_series if d > h]
        if candidates_after:
            post_holiday_dates.add(min(candidates_after))

    df["is_pre_holiday"]  = df["ds"].isin(pre_holiday_dates).astype(int)
    df["is_post_holiday"] = df["ds"].isin(post_holiday_dates).astype(int)

    # ── ISO calendar features ─────────────────────────────────────────────────
    iso = df["ds"].dt.isocalendar()
    df["iso_year"]    = iso["year"].astype(int)
    df["week_of_year"] = iso["week"].astype(int)
    df["day_of_week"] = df["ds"].dt.dayofweek      # 0 = Monday
    df["month"]       = df["ds"].dt.month

    # ── Rolling statistics (shifted by 1 to avoid leakage) ───────────────────
    # rolling_N = mean of the preceding N workdays of y
    for window in (10, 20, 30):
        df[f"rolling_{window}"] = (
            df["y"].shift(1).rolling(window, min_periods=1).mean()
        )

    # ── Drop leading NaNs (first row has no rolling history at all) ───────────
    # With min_periods=1 there are no NaNs, but keep a note here for clarity.
    df = df.reset_index(drop=True)

    log.info(
        f"[{direction}] Feature columns: {list(df.columns)}  —  shape {df.shape}"
    )

    df.to_parquet(processed_data.path + ".parquet", index=False)
    log.info(f"[{direction}] Written to {processed_data.path}.parquet")
