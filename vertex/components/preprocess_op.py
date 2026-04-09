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

    _debug_log = []
    _debug_gcs_path = "gs://csb-reg-euw3-forecast-data-dev/debug/preprocess_op_debug.txt"

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

    p("=== preprocess_op started ===")

    # ── Load raw data ─────────────────────────────────────────────────────────
    df = pd.read_parquet(raw_data.path + ".parquet")
    p(f"loaded {len(df)} rows | columns: {list(df.columns)}")

    # ── Filter by warehouse ───────────────────────────────────────────────────
    if "warehouse_id" not in df.columns:
        raise ValueError(f"Column 'warehouse_id' not found. Available: {list(df.columns)}")
    df = df[df["warehouse_id"] == warehouse_id].copy()
    p(f"after warehouse_id filter ('{warehouse_id}'): {len(df)} rows")
    if len(df) == 0:
        raise ValueError(f"No rows remain after filtering warehouse_id == '{warehouse_id}'")

    # ── Rename and parse ──────────────────────────────────────────────────────
    if date_column not in df.columns:
        raise ValueError(f"date_column '{date_column}' not found. Available: {list(df.columns)}")
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found. Available: {list(df.columns)}")

    df = df.rename(columns={date_column: "ds", target_column: "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    p(f"raw y stats: min={df['y'].min():.2f}  max={df['y'].max():.2f}  "
      f"mean={df['y'].mean():.2f}  zeros={int((df['y'] == 0).sum())}  "
      f"nulls={int(df['y'].isna().sum())}")
    p(f"raw date range: {df['ds'].min()} – {df['ds'].max()}  "
      f"unique dates: {df['ds'].dt.normalize().nunique()}")
    p(f"raw sample hours: {df['ds'].dt.hour.value_counts().head(5).to_dict()}")

    # ── 5-hour backward shift ─────────────────────────────────────────────────
    # Activity between 00:00 and 04:59 belongs to the previous business day.
    df["ds"] = df["ds"] - pd.Timedelta(hours=5)
    p("applied -5 h shift to datetimes")

    # ── Aggregate to daily (sum) ──────────────────────────────────────────────
    df["ds"] = df["ds"].dt.normalize()
    daily = df.groupby("ds")["y"].sum().reset_index()
    daily = daily.sort_values("ds").reset_index(drop=True)
    p(f"aggregated to {len(daily)} daily rows "
      f"(date range: {daily['ds'].min().date()} – {daily['ds'].max().date()})")
    p(f"daily y stats: min={daily['y'].min():.2f}  max={daily['y'].max():.2f}  "
      f"mean={daily['y'].mean():.2f}  zeros={int((daily['y'] == 0).sum())}")

    flush_debug_to_gcs()  # checkpoint: aggregation done

    # ── Filter by data_start_date ─────────────────────────────────────────────
    start = pd.Timestamp(data_start_date)
    before_count = len(daily)
    daily = daily[daily["ds"] >= start].reset_index(drop=True)
    p(f"filtered to >= {data_start_date}: kept {len(daily)} / {before_count} rows")
    if len(daily) == 0:
        raise ValueError(f"No rows remain after filtering ds >= '{data_start_date}'")

    # ── Drop weekends (warehouse operates Mon-Fri only) ──────────────────────
    before_count = len(daily)
    daily = daily[daily["ds"].dt.dayofweek < 5].reset_index(drop=True)
    p(f"dropped weekends: kept {len(daily)} / {before_count} rows")
    p(f"after weekend drop: zeros={int((daily['y'] == 0).sum())} / {len(daily)}  "
      f"({100 * (daily['y'] == 0).sum() / max(len(daily), 1):.1f}%)")

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
    non_holiday_days = daily.loc[daily["is_holiday"] == 0, "ds"].tolist()
    for h in fr_holidays_set:
        before = [d for d in non_holiday_days if d < h]
        after  = [d for d in non_holiday_days if d > h]
        if before:
            pre_holiday_dates.add(max(before))
        if after:
            post_holiday_dates.add(min(after))

    daily["is_pre_holiday"]  = daily["ds"].isin(pre_holiday_dates).astype(int)
    daily["is_post_holiday"] = daily["ds"].isin(post_holiday_dates).astype(int)
    p(f"holiday flags — holidays: {int(daily['is_holiday'].sum())}, "
      f"pre: {int(daily['is_pre_holiday'].sum())}, "
      f"post: {int(daily['is_post_holiday'].sum())}")

    # ── ISO calendar features ─────────────────────────────────────────────────
    iso = daily["ds"].dt.isocalendar()
    daily["iso_year"]     = iso["year"].astype(int)
    daily["week_of_year"] = iso["week"].astype(int)
    daily["day_of_week"]  = daily["ds"].dt.dayofweek   # 0 = Monday
    daily["month"]        = daily["ds"].dt.month

    # ── Rolling statistics (shifted by 1, holidays excluded) ────────────────
    # Use only non-holiday volume so that near-zero holiday days don't dilute
    # the rolling mean which represents the normal workday demand level.
    workday_y = daily["y"].where(daily["is_holiday"] == 0)
    for window in (10, 20, 30):
        daily[f"rolling_{window}"] = (
            workday_y.shift(1).rolling(window, min_periods=1).mean()
        )

    p(f"preprocessing complete. shape: {daily.shape}  columns: {list(daily.columns)}")
    p(f"final y stats: min={daily['y'].min():.2f}  max={daily['y'].max():.2f}  "
      f"mean={daily['y'].mean():.2f}  zeros={int((daily['y'] == 0).sum())}")
    daily.to_parquet(processed_data.path + ".parquet", index=False)
    p("=== preprocess_op complete ===")
    flush_debug_to_gcs()
