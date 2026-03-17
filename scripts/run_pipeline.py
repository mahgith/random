"""
PIPELINE RUNNER
===============
Compiles and submits the forecasting pipeline for a specific direction.

Usage:
    python scripts/run_pipeline.py --direction inbound
    python scripts/run_pipeline.py --direction outbound
    python scripts/run_pipeline.py --direction inbound --compile-only
    python scripts/run_pipeline.py --direction inbound --dry-run

Prerequisites:
    1. Fill in configs/pipeline_config.yaml:
       - gcp.project_id, gcp.artifact_bucket
       - directions.inbound.bq_tables  (replace PLACEHOLDER values)
       - directions.inbound.bq_columns (replace PLACEHOLDER values)
       - same for outbound
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

from configs.settings import settings, build_pipeline_params


def compile_pipeline(output_path: str) -> None:
    from kfp.v2.compiler import Compiler
    from pipelines.pipeline.forecasting_pipeline import forecasting_pipeline

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Compiler().compile(pipeline_func=forecasting_pipeline, package_path=output_path)
    print(f"[✓] Compiled to: {output_path}")


def submit_pipeline(compiled_path: str, direction: str) -> None:
    from google.cloud import aiplatform

    params = build_pipeline_params(direction)

    aiplatform.init(
        project=settings.PROJECT_ID,
        location=settings.REGION,
        staging_bucket=settings.ARTIFACT_BUCKET,
        experiment=settings.EXPERIMENT_NAME,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    job_display_name = f"{settings.PIPELINE_NAME}-{direction}-{timestamp}"

    print(f"[→] Submitting: {job_display_name}")
    print(f"    Project:    {settings.PROJECT_ID}")
    print(f"    Region:     {settings.REGION}")
    print(f"    Direction:  {direction}")
    print(f"    Bucket:     {settings.ARTIFACT_BUCKET}")

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
        f"?project={settings.PROJECT_ID}"
    )
    print(f"\n[✓] Pipeline submitted!")
    print(f"    Monitor at: {console_url}")
    print(f"    Job name:   {job_display_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile and submit the package-volume forecasting pipeline"
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
        default=settings.COMPILED_PIPELINE_PATH,
        help="Path for the compiled pipeline JSON",
    )
    args = parser.parse_args()

    if args.dry_run:
        import json
        params = build_pipeline_params(args.direction)
        print(f"=== DRY RUN — {args.direction} ===")
        print(json.dumps(params, indent=2))
        print("\nUpdate configs/pipeline_config.yaml to change these values.")
        return

    compile_pipeline(args.output_path)

    if not args.compile_only:
        submit_pipeline(args.output_path, args.direction)


if __name__ == "__main__":
    main()
