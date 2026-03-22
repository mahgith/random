"""
PREPROCESSING COMPONENT
=======================
Reads the raw daily time series from BigQuery (written by data_ingestion_op),
adds calendar features, French public-holiday flags, and rolling statistics,
then writes the enriched DataFrame back to BigQuery.

This is part of Pipeline 1 (data prep — runs daily).
Pipeline 2 (training) reads the output BQ table directly.

Input BQ table:  project.bq_raw_dataset.bq_raw_table           (ds, y)
Output BQ table: project.bq_output_dataset.bq_output_table     (ds, y, + features)
"""

from kfp.dsl import component

_FORECASTING_IMAGE = "europe-west1-docker.pkg.dev/your-gcp-project-id/ml-images/forecasting:latest"


@component(base_image=_FORECASTING_IMAGE)
def preprocessing_op(
    direction: str,
    project_id: str,
    location: str,
    bq_raw_dataset: str,       # dataset written by data_ingestion_op
    bq_raw_table: str,         # table written by data_ingestion_op
    bq_output_dataset: str,    # dataset to write enriched series into
    bq_output_table: str,      # table to write enriched series into
):
    """Add calendar, holiday, and rolling features to the daily time series."""
    import structlog
    import pandas as pd
    import holidays as hol_lib
    from google.cloud import bigquery
    from common.core.logger import get_logger

    logger = get_logger("preprocessing")
    structlog.contextvars.bind_contextvars(
        direction=direction,
        project_id=project_id,
        source=f"{project_id}.{bq_raw_dataset}.{bq_raw_table}",
        destination=f"{project_id}.{bq_output_dataset}.{bq_output_table}",
    )

    client = bigquery.Client(project=project_id, location=location)

    # ── Load from BigQuery ────────────────────────────────────────────────────
    raw_table = f"`{project_id}.{bq_raw_dataset}.{bq_raw_table}`"
    df = client.query(f"SELECT ds, y FROM {raw_table} ORDER BY ds").to_dataframe()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    logger.info("Loaded raw series from BigQuery", rows=len(df))

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

    # ── Write to BigQuery ─────────────────────────────────────────────────────
    full_table_id = f"{project_id}.{bq_output_dataset}.{bq_output_table}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(df, full_table_id, job_config=job_config).result()
    logger.info("Written to BigQuery", table=full_table_id, rows=len(df))
