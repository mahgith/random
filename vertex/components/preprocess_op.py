"""
PREPROCESS COMPONENT
====================
Step 2 — Transforms raw CSV data into a daily time series with all
features required by the 3-layer model.

What it does
------------
1. Filters to the specified warehouse_id (column 'warehouse_id')
2. Renames `date_column` → ds,  `target_column` → y
3. Applies a 5-hour backward shift to datetimes so that activity between
   midnight and 05:00 is attributed to the previous calendar day
4. Aggregates to daily resolution (sum) — handles hourly or sub-daily input
5. Adds French public-holiday flags: is_holiday, is_pre_holiday, is_post_holiday
6. Adds ISO calendar features: week_of_year, day_of_week, month, iso_year
7. Adds rolling statistics (shifted by 1 to prevent leakage):
       rolling_10, rolling_20, rolling_30

Output schema (parquet):
    ds               datetime64[ns]
    y                float64          — daily target value
    is_holiday       int64
    is_pre_holiday   int64
    is_post_holiday  int64
    week_of_year     int64
    day_of_week      int64            0 = Monday
    month            int64
    iso_year         int64
    rolling_10       float64
    rolling_20       float64
    rolling_30       float64
"""

from kfp.dsl import component, Input, Output, Dataset

_ML_TRAINING_IMAGE = "europe-west3-docker.pkg.dev/prj-ceva-gr-wkfc-adapt-dev/gr-forecast/ml-training:latest"


@component(base_image=_ML_TRAINING_IMAGE)
def preprocess_op(
    raw_data: Input[Dataset],
    date_column: str,
    target_column: str,
    warehouse_id: str,
    data_start_date: str,
    processed_data: Output[Dataset],
):
    """Add calendar, holiday, and rolling features to the daily time series."""
    import logging
    import pandas as pd
    import holidays as hol_lib

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    # ── Load raw data ─────────────────────────────────────────────────────────
    df = pd.read_parquet(raw_data.path + ".parquet")
    logger.info("Loaded %d rows. Columns: %s", len(df), list(df.columns))

    # ── Filter by warehouse ───────────────────────────────────────────────────
    if "warehouse_id" not in df.columns:
        raise ValueError(f"Column 'warehouse_id' not found. Available: {list(df.columns)}")
    df = df[df["warehouse_id"] == warehouse_id].copy()
    logger.info("After warehouse_id filter ('%s'): %d rows", warehouse_id, len(df))
    if len(df) == 0:
        raise ValueError(f"No rows remain after filtering warehouse_id == '{warehouse_id}'")

    # ── Rename and parse ──────────────────────────────────────────────────────
    if date_column not in df.columns:
        raise ValueError(f"date_column '{date_column}' not found. Available: {list(df.columns)}")
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found. Available: {list(df.columns)}")

    df = df.rename(columns={date_column: "ds", target_column: "y"})
    df["ds"] = pd.to_datetime(df["ds"])

    # ── 5-hour backward shift ─────────────────────────────────────────────────
    # Activity between 00:00 and 04:59 belongs to the previous business day.
    df["ds"] = df["ds"] - pd.Timedelta(hours=5)
    logger.info("Applied -5 h shift to datetimes")

    # ── Aggregate to daily (sum) ──────────────────────────────────────────────
    df["ds"] = df["ds"].dt.normalize()
    daily = df.groupby("ds")["y"].sum().reset_index()
    daily = daily.sort_values("ds").reset_index(drop=True)
    logger.info("Aggregated to %d daily rows (date range: %s – %s)",
                len(daily), daily["ds"].min().date(), daily["ds"].max().date())

    # ── Filter by data_start_date ─────────────────────────────────────────────
    start = pd.Timestamp(data_start_date)
    before = len(daily)
    daily = daily[daily["ds"] >= start].reset_index(drop=True)
    logger.info("Filtered to >= %s: kept %d / %d rows", data_start_date, len(daily), before)
    if len(daily) == 0:
        raise ValueError(f"No rows remain after filtering ds >= '{data_start_date}'")

    # ── French public holidays ────────────────────────────────────────────────
    years = daily["ds"].dt.year.unique().tolist()
    fr_holidays_set: set = set()
    for yr in years:
        fr_holidays_set.update(
            pd.Timestamp(d) for d in hol_lib.France(years=yr).keys()
        )

    daily["is_holiday"] = daily["ds"].isin(fr_holidays_set).astype(int)

    pre_holiday_dates: set = set()
    post_holiday_dates: set = set()
    workdays = daily["ds"].tolist()
    for h in fr_holidays_set:
        before = [d for d in workdays if d < h]
        after  = [d for d in workdays if d > h]
        if before:
            pre_holiday_dates.add(max(before))
        if after:
            post_holiday_dates.add(min(after))

    daily["is_pre_holiday"]  = daily["ds"].isin(pre_holiday_dates).astype(int)
    daily["is_post_holiday"] = daily["ds"].isin(post_holiday_dates).astype(int)
    logger.info("Holiday flags — holidays: %d, pre: %d, post: %d",
                int(daily["is_holiday"].sum()),
                int(daily["is_pre_holiday"].sum()),
                int(daily["is_post_holiday"].sum()))

    # ── ISO calendar features ─────────────────────────────────────────────────
    iso = daily["ds"].dt.isocalendar()
    daily["iso_year"]     = iso["year"].astype(int)
    daily["week_of_year"] = iso["week"].astype(int)
    daily["day_of_week"]  = daily["ds"].dt.dayofweek   # 0 = Monday
    daily["month"]        = daily["ds"].dt.month

    # ── Rolling statistics (shifted by 1 to avoid leakage) ───────────────────
    for window in (10, 20, 30):
        daily[f"rolling_{window}"] = (
            daily["y"].shift(1).rolling(window, min_periods=1).mean()
        )

    logger.info("Preprocessing complete. Shape: %s. Columns: %s",
                daily.shape, list(daily.columns))
    daily.to_parquet(processed_data.path + ".parquet", index=False)
