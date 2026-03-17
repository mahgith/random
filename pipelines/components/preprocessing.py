"""
PREPROCESSING COMPONENT
=======================
Adds calendar features, French public-holiday flags, and rolling
statistics to the raw daily time series produced by data_ingestion_op.

This matches the feature engineering cells in modeling_inbound.ipynb /
modeling_outbound.ipynb.

Input:  raw_dataset    — daily workday series [ds, y]
Output: processed_data — enriched DataFrame with all features used by
                         training and evaluation components
"""

from kfp.v2.dsl import component, Input, Output, Dataset

_FORECASTING_IMAGE = "europe-west1-docker.pkg.dev/your-gcp-project-id/ml-images/forecasting:latest"


@component(base_image=_FORECASTING_IMAGE)
def preprocessing_op(
    direction: str,
    raw_dataset: Input[Dataset],
    processed_data: Output[Dataset] = None,  # type: ignore[assignment]
):
    """Add calendar, holiday, and rolling features to the daily time series."""
    import structlog
    import pandas as pd
    import holidays as hol_lib
    from common.core.logger import get_logger

    logger = get_logger("preprocessing")
    structlog.contextvars.bind_contextvars(direction=direction)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_parquet(raw_dataset.path + ".parquet")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    logger.info("Loaded raw series", rows=len(df))

    # ── French public holidays ────────────────────────────────────────────────
    years = df["ds"].dt.year.unique().tolist()
    fr_holidays = set()
    for yr in years:
        fr_holidays.update(hol_lib.France(years=yr).keys())

    holiday_dates = pd.to_datetime(sorted(fr_holidays))
    df["is_holiday"] = df["ds"].dt.date.astype("datetime64[ns]").isin(
        holiday_dates
    ).astype(int)

    holiday_set        = set(holiday_dates.normalize())
    pre_holiday_dates  = set()
    post_holiday_dates = set()
    workday_series     = df["ds"].tolist()

    for h in holiday_set:
        before = [d for d in workday_series if d < h]
        after  = [d for d in workday_series if d > h]
        if before:
            pre_holiday_dates.add(max(before))
        if after:
            post_holiday_dates.add(min(after))

    df["is_pre_holiday"]  = df["ds"].isin(pre_holiday_dates).astype(int)
    df["is_post_holiday"] = df["ds"].isin(post_holiday_dates).astype(int)

    logger.info(
        "Holiday flags added",
        holidays=int(df["is_holiday"].sum()),
        pre_holidays=int(df["is_pre_holiday"].sum()),
        post_holidays=int(df["is_post_holiday"].sum()),
    )

    # ── ISO calendar features ─────────────────────────────────────────────────
    iso = df["ds"].dt.isocalendar()
    df["iso_year"]     = iso["year"].astype(int)
    df["week_of_year"] = iso["week"].astype(int)
    df["day_of_week"]  = df["ds"].dt.dayofweek   # 0 = Monday
    df["month"]        = df["ds"].dt.month

    # ── Rolling statistics (shifted by 1 to avoid leakage) ───────────────────
    for window in (10, 20, 30):
        df[f"rolling_{window}"] = (
            df["y"].shift(1).rolling(window, min_periods=1).mean()
        )

    df = df.reset_index(drop=True)

    logger.info("Preprocessing complete", columns=list(df.columns), shape=list(df.shape))

    df.to_parquet(processed_data.path + ".parquet", index=False)
    logger.info("Written to artifact", path=processed_data.path)
