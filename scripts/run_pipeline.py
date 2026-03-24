"""
PIPELINE RUNNER
===============
Compiles and submits either pipeline for a specific direction.

Pipelines
---------
  data_prep    — runs daily when new data arrives
                 GCS CSV → BQ raw → BQ processed features
  forecasting  — runs monthly or on performance drop
                 Training → evaluation → champion/challenger → refit → registration

Usage:
    # Pipeline 1 — data prep
    python scripts/run_pipeline.py --pipeline data_prep    --direction inbound
    python scripts/run_pipeline.py --pipeline data_prep    --direction outbound

    # Pipeline 2 — training
    python scripts/run_pipeline.py --pipeline forecasting  --direction inbound
    python scripts/run_pipeline.py --pipeline forecasting  --direction outbound

    # Helpers
    python scripts/run_pipeline.py --pipeline forecasting  --direction inbound --compile-only
    python scripts/run_pipeline.py --pipeline forecasting  --direction inbound --dry-run

Prerequisites:
    1. Fill in parameters/inbound/params_v1.yaml and parameters/outbound/params_v1.yaml:
       - infra.project_id, infra.artifact_bucket, infra.location
       - bq_tables  (replace PLACEHOLDER dataset/table values)
       - bq_columns (replace PLACEHOLDER column name values)
       - bq_ingested_dataset/table, bq_processed_dataset/table
    2. Authenticate:  gcloud auth application-default login
    3. Dependencies:  pip install -r requirements.txt
    4. Enable APIs:
         gcloud services enable aiplatform.googleapis.com
         gcloud services enable bigquery.googleapis.com
         gcloud services enable storage.googleapis.com
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import settings, build_data_prep_params, build_forecasting_params

PIPELINES = {
    "data_prep": {
        "module":    "pipelines.pipeline.data_prep_pipeline",
        "func":      "data_prep_pipeline",
        "compiled":  settings.COMPILED_DATA_PREP_PATH,
        "params_fn": build_data_prep_params,
    },
    "forecasting": {
        "module":    "pipelines.pipeline.forecasting_pipeline",
        "func":      "forecasting_pipeline",
        "compiled":  settings.COMPILED_FORECASTING_PATH,
        "params_fn": build_forecasting_params,
    },
}


def compile_pipeline(pipeline: str, output_path: str) -> None:
    import importlib
    from kfp.compiler import Compiler

    spec      = PIPELINES[pipeline]
    mod       = importlib.import_module(spec["module"])
    func      = getattr(mod, spec["func"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Compiler().compile(pipeline_func=func, package_path=output_path)
    print(f"[✓] Compiled to: {output_path}")


def submit_pipeline(pipeline: str, compiled_path: str, direction: str) -> None:
    from google.cloud import aiplatform

    params = PIPELINES[pipeline]["params_fn"](direction)

    aiplatform.init(
        project=params["project_id"],
        location=params.get("region", params.get("location")),
        staging_bucket=params.get("artifact_bucket"),
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    job_display_name = f"{settings.PIPELINE_NAME}-{pipeline}-{direction}-{timestamp}"

    print(f"[→] Submitting: {job_display_name}")
    print(f"    Pipeline:   {pipeline}")
    print(f"    Project:    {params['project_id']}")
    print(f"    Location:   {params.get('region', params.get('location'))}")
    print(f"    Direction:  {direction}")

    job = aiplatform.PipelineJob(
        display_name=job_display_name,
        template_path=compiled_path,
        job_id=job_display_name,
        parameter_values=params,
        enable_caching=True,
    )

    job.submit(service_account=None)

    console_url = (
        f"https://console.cloud.google.com/vertex-ai/pipelines/runs"
        f"?project={params['project_id']}"
    )
    print(f"\n[✓] Pipeline submitted!")
    print(f"    Monitor at: {console_url}")
    print(f"    Job name:   {job_display_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile and submit a package-volume pipeline"
    )
    parser.add_argument(
        "--pipeline",
        choices=tuple(PIPELINES),
        required=True,
        help="Which pipeline to run: 'data_prep' (daily) or 'forecasting' (monthly)",
    )
    parser.add_argument(
        "--direction",
        choices=("inbound", "outbound"),
        required=True,
        help="Which direction to run: 'inbound' or 'outbound'",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile the pipeline JSON, do not submit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and exit without compiling or submitting",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Override path for the compiled pipeline JSON",
    )
    args = parser.parse_args()

    output_path = args.output_path or PIPELINES[args.pipeline]["compiled"]

    if args.dry_run:
        import json
        params = PIPELINES[args.pipeline]["params_fn"](args.direction)
        print(f"=== DRY RUN — {args.pipeline} / {args.direction} ===")
        print(json.dumps(params, indent=2))
        print(f"\nUpdate parameters/{args.direction}/params_v1.yaml to change these values.")
        return

    compile_pipeline(args.pipeline, output_path)

    if not args.compile_only:
        submit_pipeline(args.pipeline, output_path, args.direction)


if __name__ == "__main__":
    main()
