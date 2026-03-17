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
Update parameters/inbound/params_v1.yaml or parameters/outbound/params_v1.yaml.
That is the only file you need to edit — no changes needed here.
"""

from kfp.v2.dsl import component, Output, Dataset

# Custom image — eliminates the pip install overhead at every pipeline run.
# Build with: make build-push  (from docker/forecasting/Makefile)
# Update this URI after pushing to Artifact Registry.
_FORECASTING_IMAGE = "europe-west1-docker.pkg.dev/your-gcp-project-id/ml-images/forecasting:latest"


@component(base_image=_FORECASTING_IMAGE)
def data_ingestion_op(
    direction: str,           # "inbound" or "outbound"
    project_id: str,
    bq_tables_json: str,      # JSON list of {dataset, table, date_from?, date_to?}
    bq_columns_json: str,     # JSON dict mapping logical name → actual BQ column name
    raw_dataset: Output[Dataset] = None,  # type: ignore[assignment]
):
    """Ingest raw warehouse scan data from BigQuery and write clean daily series."""
    import json
    import structlog
    import pandas as pd
    from google.cloud import bigquery
    from common.core.logger import get_logger

    logger = get_logger("data-ingestion")
    structlog.contextvars.bind_contextvars(direction=direction, project_id=project_id)

    # ── Parse parameters ─────────────────────────────────────────────────────
    tables = json.loads(bq_tables_json)
    cols   = json.loads(bq_columns_json)

    for key, val in cols.items():
        if "PLACEHOLDER" in val:
            raise ValueError(
                f"Column '{key}' is still a PLACEHOLDER ('{val}'). "
                "Update bq_columns in parameters/{direction}/params_v1.yaml."
            )
    for tbl in tables:
        for field in ("dataset", "table"):
            if "PLACEHOLDER" in tbl.get(field, ""):
                raise ValueError(
                    f"BQ {field} is still a PLACEHOLDER ('{tbl[field]}'). "
                    "Update bq_tables in parameters/{direction}/params_v1.yaml."
                )

    col_pkg   = cols["package_id"]
    col_ts    = cols["scan_timestamp"]
    col_units = cols["packing_units"]

    # ── Build UNION ALL across source tables ─────────────────────────────────
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
                CAST({col_pkg}   AS STRING)    AS package_id,
                CAST({col_ts}    AS TIMESTAMP) AS scan_timestamp,
                CAST({col_units} AS FLOAT64)   AS packing_units
            FROM `{project_id}.{tbl['dataset']}.{tbl['table']}`
            {where}
        """)

    union_sql = "\nUNION ALL\n".join(union_parts)

    query = f"""
    WITH raw AS ({union_sql}),
    adjusted AS (
        SELECT
            package_id,
            TIMESTAMP_SUB(scan_timestamp, INTERVAL 5 HOUR) AS local_ts,
            packing_units
        FROM raw
    ),
    ranked AS (
        SELECT
            package_id,
            DATE(local_ts)  AS warehouse_day,
            packing_units,
            ROW_NUMBER() OVER (PARTITION BY package_id ORDER BY local_ts ASC) AS rn
        FROM adjusted
    ),
    deduped AS (
        SELECT warehouse_day, packing_units FROM ranked WHERE rn = 1
    )
    SELECT
        warehouse_day      AS ds,
        SUM(packing_units) AS y
    FROM deduped
    GROUP BY warehouse_day
    ORDER BY ds
    """

    logger.info("Running BQ query", num_tables=len(tables))
    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()
    logger.info("BQ query complete", raw_days=len(df))

    df["ds"] = pd.to_datetime(df["ds"])
    df["y"]  = df["y"].astype(float)

    # ── Weekend folding ───────────────────────────────────────────────────────
    dow = df["ds"].dt.dayofweek
    df.loc[dow == 5, "ds"] = df.loc[dow == 5, "ds"] - pd.Timedelta(days=1)
    df.loc[dow == 6, "ds"] = df.loc[dow == 6, "ds"] + pd.Timedelta(days=1)

    df = df.groupby("ds", as_index=False)["y"].sum()
    df = df.sort_values("ds").reset_index(drop=True)
    df = df[df["ds"].dt.dayofweek < 5].reset_index(drop=True)

    logger.info(
        "Series ready",
        workdays=len(df),
        date_min=str(df["ds"].min().date()),
        date_max=str(df["ds"].max().date()),
        total_y=round(float(df["y"].sum()), 0),
    )

    df.to_parquet(raw_dataset.path + ".parquet", index=False)
    logger.info("Written to artifact", path=raw_dataset.path)
