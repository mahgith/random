"""
DATA INGESTION COMPONENT
========================
Pulls raw package scan data from BigQuery, applies the same transformations
as the inbound.ipynb / outbound.ipynb notebooks, and writes a clean
daily workday time series to GCS.

Transformations (matching notebooks exactly):
  1. UTC-5 h offset:  warehouse_day = floor((scan_timestamp - 5 h), 'day')
  2. Deduplication:   keep the earliest scan per package_id
  3. Aggregation:     sum packing_units per warehouse_day
  4. Weekend folding: Saturday → Friday,  Sunday → Monday
  5. Re-aggregate:    sum again in case Sat/Sun land on the same workday
  6. Filter:          keep Mon-Fri only

Output columns:  ds (date), y (float)

HOW TO PLUG IN YOUR BQ TABLES
------------------------------
Search for "PLACEHOLDER" in this file and in pipeline_config.yaml.
Two things to replace:
  a) Table names: bq_tables_json parameter (set in pipeline_config.yaml)
  b) Column names: bq_columns_json parameter (set in pipeline_config.yaml)

The SQL in this component uses the column names you pass via bq_columns_json,
so you only need to update the config — not the component code.
"""

from kfp.v2.dsl import component, Output, Dataset


@component(
    packages_to_install=[
        "pandas>=2.0.0",
        "pyarrow",
        "google-cloud-bigquery>=3.0.0",
        "db-dtypes",
    ],
    base_image="python:3.10-slim",
)
def data_ingestion_op(
    direction: str,           # "inbound" or "outbound"
    project_id: str,
    bq_tables_json: str,      # JSON list of {dataset, table, date_from?, date_to?}
    bq_columns_json: str,     # JSON dict mapping logical name → actual BQ column name
    raw_dataset: Output[Dataset] = None,  # type: ignore[assignment]
):
    """Ingest raw warehouse scan data from BigQuery and write clean daily series."""
    import json
    import logging
    import pandas as pd
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # ── Parse parameters ─────────────────────────────────────────────────────
    tables = json.loads(bq_tables_json)
    cols = json.loads(bq_columns_json)

    # Validate that PLACEHOLDER values have been replaced
    for key, val in cols.items():
        if "PLACEHOLDER" in val:
            raise ValueError(
                f"Column name '{key}' is still a PLACEHOLDER ('{val}'). "
                "Update bq_columns in pipeline_config.yaml."
            )
    for tbl in tables:
        for field in ("dataset", "table"):
            if "PLACEHOLDER" in tbl.get(field, ""):
                raise ValueError(
                    f"BQ {field} is still a PLACEHOLDER ('{tbl[field]}'). "
                    "Update bq_tables in pipeline_config.yaml."
                )

    col_pkg   = cols["package_id"]
    col_ts    = cols["scan_timestamp"]
    col_units = cols["packing_units"]

    # ── Build UNION ALL across source tables ─────────────────────────────────
    # Each source table may have an optional date_from / date_to pre-filter.
    # Filtering in SQL before UNION reduces bytes scanned.
    union_parts = []
    for tbl in tables:
        filters = []
        if tbl.get("date_from"):
            filters.append(
                f"DATE(TIMESTAMP_SUB({col_ts}, INTERVAL 5 HOUR)) >= '{tbl['date_from']}'"
            )
        if tbl.get("date_to"):
            filters.append(
                f"DATE(TIMESTAMP_SUB({col_ts}, INTERVAL 5 HOUR)) <= '{tbl['date_to']}'"
            )
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        union_parts.append(f"""
            SELECT
                CAST({col_pkg}   AS STRING)  AS package_id,
                CAST({col_ts}    AS TIMESTAMP) AS scan_timestamp,
                CAST({col_units} AS FLOAT64) AS packing_units
            FROM `{project_id}.{tbl['dataset']}.{tbl['table']}`
            {where}
        """)

    union_sql = "\nUNION ALL\n".join(union_parts)

    # ── Main query: offset → dedup → aggregate ────────────────────────────────
    # Steps 1-3 happen entirely in BQ for efficiency.
    # Weekend folding (steps 4-5) happens in pandas after the query.
    query = f"""
    WITH raw AS (
        {union_sql}
    ),
    adjusted AS (
        -- Step 1: UTC-5 h offset (warehouse local time)
        SELECT
            package_id,
            TIMESTAMP_SUB(scan_timestamp, INTERVAL 5 HOUR) AS local_ts,
            packing_units
        FROM raw
    ),
    ranked AS (
        -- Step 2: Rank by first scan per package (deduplication)
        SELECT
            package_id,
            DATE(local_ts)  AS warehouse_day,
            packing_units,
            ROW_NUMBER() OVER (PARTITION BY package_id ORDER BY local_ts ASC) AS rn
        FROM adjusted
    ),
    deduped AS (
        SELECT warehouse_day, packing_units
        FROM ranked
        WHERE rn = 1
    )
    -- Step 3: Aggregate by day
    SELECT
        warehouse_day          AS ds,
        SUM(packing_units)     AS y
    FROM deduped
    GROUP BY warehouse_day
    ORDER BY ds
    """

    log.info(f"[{direction}] Running BQ query across {len(tables)} table(s) ...")
    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()
    log.info(f"[{direction}] Raw result: {len(df)} days from BQ")

    df["ds"] = pd.to_datetime(df["ds"])
    df["y"]  = df["y"].astype(float)

    # ── Step 4-5: Weekend folding ─────────────────────────────────────────────
    # Saturday (5) → Friday (-1 day)
    # Sunday  (6) → Monday (+1 day)
    dow = df["ds"].dt.dayofweek
    df.loc[dow == 5, "ds"] = df.loc[dow == 5, "ds"] - pd.Timedelta(days=1)
    df.loc[dow == 6, "ds"] = df.loc[dow == 6, "ds"] + pd.Timedelta(days=1)

    # Re-aggregate: Sat and Fri (or Sun and Mon) may now be on the same date
    df = df.groupby("ds", as_index=False)["y"].sum()
    df = df.sort_values("ds").reset_index(drop=True)

    # ── Step 6: Keep Mon-Fri only ─────────────────────────────────────────────
    df = df[df["ds"].dt.dayofweek < 5].reset_index(drop=True)

    log.info(
        f"[{direction}] Final series: {len(df)} workdays "
        f"from {df['ds'].min().date()} to {df['ds'].max().date()}, "
        f"total y = {df['y'].sum():,.0f}"
    )

    df.to_parquet(raw_dataset.path + ".parquet", index=False)
    log.info(f"[{direction}] Written to {raw_dataset.path}.parquet")
